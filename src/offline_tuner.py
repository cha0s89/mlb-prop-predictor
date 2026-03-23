"""
Offline backtest tuner for selection-policy confidence floors.

This module analyzes historical backtest results on a chronological
train/validation split and recommends updated per-prop confidence floors.
It is intentionally conservative: it will not auto-promote a candidate
unless the validation sample improves cleanly without collapsing volume.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gamma as sp_gamma, nbinom as sp_nbinom, norm as sp_norm, poisson as sp_poisson

from src.autolearn import (
    CALIBRATION_PATH,
    load_current_weights,
    rebuild_calibration_tables,
    save_weights,
    _next_version,
)
from src.backtester import DEFAULT_RESULTS_PATH, load_results, filter_nonplays
from src.predictor import get_distribution_config
from src.selection import floor_key, get_confidence_floor
from src.tail_signals import INVERSE_GOOD_PROPS


SUPPORTED_PROPS = {
    "hits",
    "total_bases",
    "pitcher_strikeouts",
    "hitter_fantasy_score",
    "home_runs",
    "runs",
    "rbis",
    "hits_runs_rbis",
    "batter_strikeouts",
    "walks",
    "singles",
    "doubles",
    "pitching_outs",
    "earned_runs",
    "walks_allowed",
    "hits_allowed",
}
DEFAULT_GRID = [round(x, 2) for x in np.arange(0.54, 0.91, 0.02)]
MIN_TRAIN_PICKS = 80
MIN_VALID_PICKS = 30
MIN_VOLUME_RETAIN = 0.85
MIN_ACCURACY_GAIN = 0.003

MODEL_TUNING_PROPS = {
    "hits",
    "total_bases",
    "pitcher_strikeouts",
    "hitter_fantasy_score",
    "home_runs",
    "runs",
    "rbis",
    "hits_runs_rbis",
    "batter_strikeouts",
    "walks",
    "singles",
    "doubles",
    "pitching_outs",
    "earned_runs",
    "walks_allowed",
    "hits_allowed",
}

OFFSET_GRIDS = {
    "hits": [round(x, 2) for x in np.arange(-0.30, 0.31, 0.05)],
    "total_bases": [round(x, 2) for x in np.arange(-0.80, 0.81, 0.10)],
    "pitcher_strikeouts": [round(x, 2) for x in np.arange(-1.00, 1.01, 0.10)],
    "hitter_fantasy_score": [round(x, 2) for x in np.arange(-3.00, 3.01, 0.25)],
    "home_runs": [round(x, 3) for x in np.arange(-0.15, 0.151, 0.025)],
    "runs": [round(x, 2) for x in np.arange(-0.60, 0.61, 0.05)],
    "rbis": [round(x, 2) for x in np.arange(-0.70, 0.71, 0.05)],
    "hits_runs_rbis": [round(x, 2) for x in np.arange(-1.50, 1.51, 0.10)],
    "batter_strikeouts": [round(x, 2) for x in np.arange(-0.80, 0.81, 0.05)],
    "walks": [round(x, 2) for x in np.arange(-0.50, 0.51, 0.05)],
    "singles": [round(x, 2) for x in np.arange(-0.80, 0.81, 0.05)],
    "doubles": [round(x, 2) for x in np.arange(-0.40, 0.41, 0.05)],
    "pitching_outs": [round(x, 2) for x in np.arange(-4.00, 4.01, 0.25)],
    "earned_runs": [round(x, 2) for x in np.arange(-1.20, 1.21, 0.10)],
    "walks_allowed": [round(x, 2) for x in np.arange(-1.00, 1.01, 0.10)],
    "hits_allowed": [round(x, 2) for x in np.arange(-2.00, 2.01, 0.10)],
}
VARIANCE_MULTIPLIERS = [0.70, 0.85, 1.0, 1.15, 1.30, 1.50]
BLEND_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
SHRINKAGE_GRID = [round(x, 2) for x in np.arange(0.50, 1.01, 0.05)]

MODEL_MIN_TRAIN_ROWS = 120
MODEL_MIN_VALID_ROWS = 40
MODEL_MIN_LOGLOSS_GAIN = 0.005
MODEL_MAX_MAE_REGRESSION = 0.02
TAIL_MIN_TRAIN_ROWS = 120
TAIL_MIN_VALID_ROWS = 40
TAIL_MEDIUM_GRID = [round(x, 2) for x in np.arange(0.08, 0.31, 0.02)]
TAIL_HIGH_GRID = [round(x, 2) for x in np.arange(0.12, 0.51, 0.02)]
TAIL_MIN_SCORE_GAIN = 1e-4
TAIL_MIN_HOLDOUT_GAIN = 1e-4


def _offset_grid_for_prop(prop_type: str, subset: pd.DataFrame, current_offset: float) -> list[float]:
    """Build a prop-specific offset search grid with a data-driven fallback."""
    manual = OFFSET_GRIDS.get(prop_type)
    if manual:
        return sorted(set(manual + [round(current_offset, 4)]))

    residual = pd.to_numeric(subset["actual"], errors="coerce") - pd.to_numeric(subset["projection"], errors="coerce")
    residual = residual.replace([np.inf, -np.inf], np.nan).dropna()
    if residual.empty:
        return [round(current_offset, 4)]

    bias = float(residual.mean())
    spread = float(max(residual.std(ddof=0), abs(bias), 0.15))
    radius = min(max(spread * 1.25, 0.25), 3.0)
    center = current_offset + bias
    grid = np.linspace(center - radius, center + radius, 11)
    return sorted(set(round(float(x), 4) for x in list(grid) + [current_offset, center]))


def _metric_or_default(metrics: dict, key: str, default: float = 1e9) -> float:
    """Return a metric value with a numeric default when missing."""
    value = metrics.get(key)
    return default if value is None else float(value)


@dataclass
class FloorCandidate:
    floor: float
    count: int
    accuracy: float
    score: float


def load_backtest_dataframe(filepath: str = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load supported backtest results as a graded DataFrame."""
    results = load_results(filepath)
    if not results:
        return pd.DataFrame()

    plays, _ = filter_nonplays(results)
    df = pd.DataFrame(plays)
    if df.empty:
        return df

    df = df[df["result"].isin(["W", "L"])].copy()
    df = df[df["prop_type"].isin(SUPPORTED_PROPS)].copy()
    if df.empty:
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["is_win"] = (df["result"] == "W").astype(int)
    df["floor_key"] = df.apply(lambda row: floor_key(row["prop_type"], row["pick"]), axis=1)
    return df


def split_backtest_dataframe(df: pd.DataFrame, train_frac: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split the backtest by date."""
    if df.empty:
        return df.copy(), df.copy()

    unique_dates = sorted(df["game_date"].dt.date.unique().tolist())
    if len(unique_dates) < 4:
        return df.copy(), df.iloc[0:0].copy()

    split_idx = max(1, int(len(unique_dates) * train_frac))
    split_idx = min(split_idx, len(unique_dates) - 1)
    train_dates = set(unique_dates[:split_idx])
    valid_dates = set(unique_dates[split_idx:])
    train_df = df[df["game_date"].dt.date.isin(train_dates)].copy()
    valid_df = df[df["game_date"].dt.date.isin(valid_dates)].copy()
    return train_df, valid_df


def split_backtest_dataframe_three_way(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    valid_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split the backtest into train/validation/holdout windows."""
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    unique_dates = sorted(df["game_date"].dt.date.unique().tolist())
    if len(unique_dates) < 9:
        train_df, valid_df = split_backtest_dataframe(df, train_frac=0.75)
        return train_df, valid_df, df.iloc[0:0].copy()

    train_idx = max(1, int(len(unique_dates) * train_frac))
    valid_idx = max(train_idx + 1, int(len(unique_dates) * (train_frac + valid_frac)))
    valid_idx = min(valid_idx, len(unique_dates) - 1)

    train_dates = set(unique_dates[:train_idx])
    valid_dates = set(unique_dates[train_idx:valid_idx])
    holdout_dates = set(unique_dates[valid_idx:])

    train_df = df[df["game_date"].dt.date.isin(train_dates)].copy()
    valid_df = df[df["game_date"].dt.date.isin(valid_dates)].copy()
    holdout_df = df[df["game_date"].dt.date.isin(holdout_dates)].copy()
    return train_df, valid_df, holdout_df


def load_model_backtest_dataframe(filepath: str = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load graded backtest rows for projection/probability tuning."""
    results = load_results(filepath)
    if not results:
        return pd.DataFrame()

    plays, _ = filter_nonplays(results)
    df = pd.DataFrame(plays)
    if df.empty:
        return df

    required = {"game_date", "prop_type", "projection", "line", "actual"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame()

    df = df[df["prop_type"].isin(MODEL_TUNING_PROPS)].copy()
    if df.empty:
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in ("projection", "line", "actual"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["projection", "line", "actual"])
    df["actual_over"] = (df["actual"] > df["line"]).astype(float)
    return df


def load_calibration_backtest_dataframe(filepath: str = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load graded backtest rows for empirical calibration-table rebuilds."""
    results = load_results(filepath)
    if not results:
        return pd.DataFrame()

    plays, _ = filter_nonplays(results)
    df = pd.DataFrame(plays)
    if df.empty:
        return df

    required = {"game_date", "prop_type", "projection", "line", "actual", "result"}
    if required - set(df.columns):
        return pd.DataFrame()

    df = df[df["result"].isin(["W", "L"])].copy()
    if df.empty:
        return df

    df = df.rename(columns={"prop_type": "stat_internal", "actual": "actual_result"})
    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in ("projection", "line", "actual_result"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["projection", "line", "actual_result"]).copy()


def _confidence_shrinkage_value(weights: dict, prop_type: str) -> float:
    """Read per-prop confidence shrinkage from weights with fallback."""
    raw = weights.get("confidence_shrinkage", 0.70)
    if isinstance(raw, dict):
        raw = raw.get(prop_type, raw.get("default", 0.70))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.70
    return max(0.30, min(1.10, value))


def _calibration_blend_value(weights: dict, prop_type: str) -> float:
    """Read per-prop empirical blend weight with metadata fallback."""
    raw = weights.get("calibration_blend_weights")
    if not raw:
        raw = weights.get("metadata", {}).get("calibration_blend_weights", {})
    try:
        return max(0.0, min(1.0, float((raw or {}).get(prop_type, 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _active_variance_value(weights: dict, prop_type: str) -> float:
    """Read the active variance ratio for a prop."""
    cfg = get_distribution_config(prop_type, weights)
    return float(cfg["var_ratio"])


def _calibration_tables() -> dict:
    """Load empirical calibration tables from disk."""
    from src.predictor import _load_calibration

    return _load_calibration()


def _empirical_probs_vectorized(projections: np.ndarray, prop_type: str, line: float) -> dict | None:
    """Vectorized approximation of predictor empirical calibration interpolation."""
    cal = _calibration_tables()
    prop_cal = cal.get(prop_type)
    if not prop_cal:
        return None

    cal_line = prop_cal.get("line", 0)
    if line > 0 and cal_line > 0 and abs(line - cal_line) > 0.01:
        return None

    points = [pt for pt in prop_cal.get("points", []) if pt.get("n", 0) >= 30]
    if not points:
        return None

    mids = np.array([float(pt["proj_mid"]) for pt in points], dtype=float)
    p_over = np.array([float(pt["p_over"]) for pt in points], dtype=float)
    p_under = np.array([float(pt["p_under"]) for pt in points], dtype=float)
    counts = np.array([float(pt["n"]) for pt in points], dtype=float)

    over = np.interp(projections, mids, p_over, left=p_over[0], right=p_over[-1])
    under = np.interp(projections, mids, p_under, left=p_under[0], right=p_under[-1])
    n_est = np.interp(projections, mids, counts, left=counts[0], right=counts[-1])
    return {
        "p_over": over,
        "p_under": under,
        "n_est": n_est,
        "is_dead_zone": np.maximum(over, under) < 0.54,
    }


def _vectorized_theoretical_probs(mu: np.ndarray, line: float, prop_type: str, weights: dict) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized distribution probabilities for a legacy backtest row set."""
    cfg = get_distribution_config(prop_type, weights)
    dist_type = cfg["dist_type"]
    var_ratio = float(cfg["var_ratio"])
    threshold = int(np.ceil(line))

    if prop_type == "pitcher_strikeouts" and dist_type == "betabinom":
        dist_type = "negbin"

    if dist_type == "gamma":
        safe_mu = np.clip(mu, 0.01, None)
        var = np.clip(safe_mu * var_ratio, 0.01, None)
        shape = (safe_mu ** 2) / var
        scale = var / np.clip(safe_mu, 0.001, None)
        p_over = 1.0 - sp_gamma.cdf(line + 0.5, shape, scale=scale)
        p_under = sp_gamma.cdf(line - 0.5, shape, scale=scale)
        return np.asarray(p_over, dtype=float), np.asarray(p_under, dtype=float)

    if dist_type == "normal":
        sigma = np.maximum(np.sqrt(np.clip(mu, 0.01, None) * var_ratio), 0.25)
        p_over = 1.0 - sp_norm.cdf(line + 0.5, loc=mu, scale=sigma)
        p_under = sp_norm.cdf(line - 0.5, loc=mu, scale=sigma)
        return np.asarray(p_over, dtype=float), np.asarray(p_under, dtype=float)

    if dist_type == "poisson":
        p_over = 1.0 - sp_poisson.cdf(threshold - 1, np.clip(mu, 0.01, None))
        return np.asarray(p_over, dtype=float), 1.0 - np.asarray(p_over, dtype=float)

    safe_mu = np.clip(mu, 0.01, None)
    if var_ratio <= 1.0:
        p_over = 1.0 - sp_poisson.cdf(threshold - 1, safe_mu)
        return np.asarray(p_over, dtype=float), 1.0 - np.asarray(p_over, dtype=float)

    var = safe_mu * var_ratio
    n_param = np.clip((safe_mu ** 2) / np.clip(var - safe_mu, 0.001, None), 0.5, None)
    p_param = np.clip(safe_mu / np.clip(var, 0.001, None), 0.001, 0.999)
    p_over = 1.0 - sp_nbinom.cdf(threshold - 1, n_param, p_param)
    return np.asarray(p_over, dtype=float), 1.0 - np.asarray(p_over, dtype=float)


def _scored_rows_for_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Score a model configuration against legacy backtest rows."""
    if df.empty:
        return df.copy()

    parts = []
    for prop_type, subset in df.groupby("prop_type"):
        work = subset.copy()
        offset = float(weights.get("prop_type_offsets", {}).get(prop_type, 0.0))
        mu = np.clip(work["projection"].astype(float).values + offset, 0.01, None)
        work["projection_adj"] = mu

        p_over, p_under = _vectorized_theoretical_probs(mu, float(work["line"].iloc[0]), prop_type, weights)

        blend = _calibration_blend_value(weights, prop_type)
        if blend > 0:
            empirical = _empirical_probs_vectorized(mu, prop_type, float(work["line"].iloc[0]))
            if empirical is not None:
                emp_weight = np.where(empirical["n_est"] >= 100, blend, blend * 0.8)
                theo_weight = 1.0 - emp_weight
                p_over = empirical["p_over"] * emp_weight + p_over * theo_weight
                p_under = empirical["p_under"] * emp_weight + p_under * theo_weight
                total = np.clip(p_over + p_under, 1e-9, None)
                p_over = p_over / total
                p_under = p_under / total

        pick_more = p_over >= p_under
        raw_prob = np.where(pick_more, p_over, p_under)
        shrink = _confidence_shrinkage_value(weights, prop_type)
        confidence = np.clip(0.50 + (raw_prob - 0.50) * shrink, 0.50, 1.0)
        actual = work["actual"].astype(float).values
        line = work["line"].astype(float).values
        is_win = np.where(pick_more, actual > line, actual < line).astype(float)

        work["pick_candidate"] = np.where(pick_more, "MORE", "LESS")
        work["confidence_candidate"] = confidence
        work["is_win_candidate"] = is_win
        parts.append(work)

    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def evaluate_model_weights(df: pd.DataFrame, weights: dict) -> dict:
    """Evaluate a full weight configuration on graded backtest rows."""
    scored = _scored_rows_for_weights(df, weights)
    if scored.empty:
        return {
            "rows": 0,
            "accuracy": None,
            "brier_score": None,
            "log_loss": None,
            "mae": None,
            "rmse": None,
            "bias": None,
            "by_prop": {},
        }

    probs = np.clip(scored["confidence_candidate"].astype(float).values, 1e-7, 1 - 1e-7)
    outcomes = scored["is_win_candidate"].astype(float).values
    proj = scored["projection_adj"].astype(float).values
    actual = scored["actual"].astype(float).values

    result = {
        "rows": int(len(scored)),
        "accuracy": float(np.mean(outcomes)),
        "brier_score": float(np.mean((probs - outcomes) ** 2)),
        "log_loss": float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))),
        "mae": float(np.mean(np.abs(proj - actual))),
        "rmse": float(np.sqrt(np.mean((proj - actual) ** 2))),
        "bias": float(np.mean(proj - actual)),
        "by_prop": {},
    }

    for prop_type, subset in scored.groupby("prop_type"):
        prop_probs = np.clip(subset["confidence_candidate"].astype(float).values, 1e-7, 1 - 1e-7)
        prop_outcomes = subset["is_win_candidate"].astype(float).values
        prop_proj = subset["projection_adj"].astype(float).values
        prop_actual = subset["actual"].astype(float).values
        result["by_prop"][prop_type] = {
            "rows": int(len(subset)),
            "accuracy": float(np.mean(prop_outcomes)),
            "brier_score": float(np.mean((prop_probs - prop_outcomes) ** 2)),
            "log_loss": float(-np.mean(prop_outcomes * np.log(prop_probs) + (1 - prop_outcomes) * np.log(1 - prop_probs))),
            "mae": float(np.mean(np.abs(prop_proj - prop_actual))),
            "rmse": float(np.sqrt(np.mean((prop_proj - prop_actual) ** 2))),
            "bias": float(np.mean(prop_proj - prop_actual)),
        }

    return result


def rebuild_backtest_calibration_tables(
    filepath: str = DEFAULT_RESULTS_PATH,
    min_per_bin: int = 40,
) -> dict:
    """Regenerate empirical calibration tables from the historical backtest."""
    df = load_calibration_backtest_dataframe(filepath)
    if df.empty:
        return {"error": "No graded rows available to rebuild calibration tables."}

    before = {}
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, encoding="utf-8") as f:
            before = json.load(f)

    rebuilt = rebuild_calibration_tables(df, min_per_bin=min_per_bin)
    after = {}
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, encoding="utf-8") as f:
            after = json.load(f)

    changed_props = sorted(set(after.keys()) - set(before.keys()))
    changed_props.extend(
        prop for prop in sorted(set(after.keys()) & set(before.keys()))
        if after.get(prop) != before.get(prop)
    )

    return {
        "backtest_path": str(filepath),
        "rows": int(len(df)),
        "props_written": sorted(after.keys()),
        "changed_props": changed_props,
        "calibration_path": str(CALIBRATION_PATH),
        "error": None,
    }


def load_tail_backtest_dataframe(filepath: str = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load backtest rows with breakout/dud probabilities for tail-label tuning."""
    results = load_results(filepath)
    if not results:
        return pd.DataFrame()

    plays, _ = filter_nonplays(results)
    df = pd.DataFrame(plays)
    if df.empty:
        return df

    required = {
        "game_date", "prop_type", "actual", "breakout_prob", "dud_prob",
        "breakout_target", "dud_target",
    }
    if required - set(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in ("actual", "breakout_prob", "dud_prob", "breakout_target", "dud_target"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["actual", "breakout_prob", "dud_prob"]).copy()
    if df.empty:
        return df

    df["actual_breakout"] = df.apply(
        lambda row: _actual_tail_event(row, kind="breakout"),
        axis=1,
    ).astype(int)
    df["actual_dud"] = df.apply(
        lambda row: _actual_tail_event(row, kind="dud"),
        axis=1,
    ).astype(int)
    return df


def _actual_tail_event(row: pd.Series, kind: str) -> bool:
    """Return whether a historical outcome satisfied the configured tail event."""
    prop_type = str(row.get("prop_type", "")).strip()
    actual = float(row.get("actual", np.nan))
    target = row.get("breakout_target" if kind == "breakout" else "dud_target")
    if pd.isna(actual) or pd.isna(target):
        return False
    if prop_type in INVERSE_GOOD_PROPS:
        return actual <= target if kind == "breakout" else actual >= target
    return actual >= target if kind == "breakout" else actual <= target


def _tail_bucket_stats(df: pd.DataFrame, prob_col: str, outcome_col: str, medium: float, high: float) -> dict:
    """Evaluate medium/high label quality for a tail-probability column."""
    actual = df[outcome_col].astype(float)
    baseline = float(actual.mean()) if len(df) else 0.0

    high_mask = df[prob_col] >= high
    med_mask = (df[prob_col] >= medium) & (df[prob_col] < high)

    def _bucket(mask: pd.Series) -> dict:
        bucket = df[mask]
        count = int(len(bucket))
        hits = int(bucket[outcome_col].sum()) if count else 0
        precision = float(bucket[outcome_col].mean()) if count else 0.0
        lift = precision / baseline if baseline > 0 else 0.0
        return {
            "count": count,
            "hits": hits,
            "precision": precision,
            "lift": lift,
            "coverage": count / len(df) if len(df) else 0.0,
            "wilson_lb": _wilson_lower_bound(hits, count) if count else 0.0,
        }

    return {
        "baseline_rate": baseline,
        "medium": _bucket(med_mask),
        "high": _bucket(high_mask),
    }


def _tail_score(stats: dict, min_high: int = 20, min_medium: int = 50) -> float:
    """Score a tail-threshold configuration, preferring high-lift stable labels."""
    high = stats["high"]
    medium = stats["medium"]
    if high["count"] < min_high or medium["count"] < min_medium:
        return -1.0

    return (
        high["wilson_lb"] * max(high["lift"], 1.0)
        + 0.35 * medium["wilson_lb"] * max(medium["lift"], 1.0)
        + 0.05 * min(high["coverage"], 0.10)
    )


def optimize_tail_signal_config(train_df: pd.DataFrame, base_weights: dict) -> dict:
    """Tune per-prop breakout/dud label cutoffs from historical backtest rows."""
    tuned_cfg = copy.deepcopy(base_weights.get("tail_signal_config", {}))
    label_thresholds_by_prop = copy.deepcopy(tuned_cfg.get("label_thresholds_by_prop", {}))
    recommendations = {}

    for prop_type, subset in train_df.groupby("prop_type"):
        if len(subset) < TAIL_MIN_TRAIN_ROWS:
            continue

        prop_rec = {}
        existing = label_thresholds_by_prop.get(prop_type, {})

        for kind, prob_col, outcome_col in (
            ("breakout", "breakout_prob", "actual_breakout"),
            ("dud", "dud_prob", "actual_dud"),
        ):
            current_medium = float(existing.get(f"{kind}_medium", 0.10 if kind == "breakout" else 0.20))
            current_high = float(existing.get(f"{kind}_high", 0.20 if kind == "breakout" else 0.35))

            best_medium = current_medium
            best_high = current_high
            best_stats = _tail_bucket_stats(subset, prob_col, outcome_col, current_medium, current_high)
            best_score = _tail_score(best_stats)

            for medium in TAIL_MEDIUM_GRID:
                for high in TAIL_HIGH_GRID:
                    if high <= medium:
                        continue
                    stats = _tail_bucket_stats(subset, prob_col, outcome_col, medium, high)
                    score = _tail_score(stats)
                    if score > best_score:
                        best_score = score
                        best_medium = medium
                        best_high = high
                        best_stats = stats

            label_thresholds_by_prop.setdefault(prop_type, {})
            label_thresholds_by_prop[prop_type][f"{kind}_medium"] = round(best_medium, 4)
            label_thresholds_by_prop[prop_type][f"{kind}_high"] = round(best_high, 4)
            prop_rec[kind] = {
                "from": {"medium": current_medium, "high": current_high},
                "to": {"medium": round(best_medium, 4), "high": round(best_high, 4)},
                "train_high_precision": round(best_stats["high"]["precision"], 4),
                "train_high_lift": round(best_stats["high"]["lift"], 4),
                "train_high_count": best_stats["high"]["count"],
            }

        recommendations[prop_type] = prop_rec

    tuned_cfg["label_thresholds_by_prop"] = label_thresholds_by_prop
    return {"tail_signal_config": tuned_cfg, "recommendations": recommendations}


def _tail_configs_differ(current_cfg: dict, candidate_cfg: dict) -> bool:
    """Return whether a tail-threshold candidate meaningfully differs from current settings."""
    return json.dumps(current_cfg or {}, sort_keys=True) != json.dumps(candidate_cfg or {}, sort_keys=True)


def _average_nonnegative(scores: list[float]) -> float:
    """Average nonnegative scores without emitting NumPy warnings for empty lists."""
    valid = [float(score) for score in scores if score >= 0]
    return float(np.mean(valid)) if valid else 0.0


def evaluate_tail_signal_config(df: pd.DataFrame, tail_cfg: dict) -> dict:
    """Evaluate a tail-signal label configuration on historical rows."""
    if df.empty:
        return {"rows": 0, "by_prop": {}, "breakout_score": 0.0, "dud_score": 0.0}

    default_labels = tail_cfg.get("label_thresholds", {}) or {}
    per_prop = tail_cfg.get("label_thresholds_by_prop", {}) or {}
    by_prop = {}
    breakout_scores = []
    dud_scores = []

    for prop_type, subset in df.groupby("prop_type"):
        cfg = dict(default_labels)
        cfg.update(per_prop.get(prop_type, {}))
        breakout_stats = _tail_bucket_stats(
            subset,
            "breakout_prob",
            "actual_breakout",
            float(cfg.get("breakout_medium", 0.10)),
            float(cfg.get("breakout_high", 0.20)),
        )
        dud_stats = _tail_bucket_stats(
            subset,
            "dud_prob",
            "actual_dud",
            float(cfg.get("dud_medium", 0.20)),
            float(cfg.get("dud_high", 0.35)),
        )
        breakout_score = _tail_score(breakout_stats)
        dud_score = _tail_score(dud_stats)
        breakout_scores.append(breakout_score)
        dud_scores.append(dud_score)
        by_prop[prop_type] = {
            "breakout_high_precision": round(breakout_stats["high"]["precision"], 4),
            "breakout_high_lift": round(breakout_stats["high"]["lift"], 4),
            "breakout_high_count": breakout_stats["high"]["count"],
            "dud_high_precision": round(dud_stats["high"]["precision"], 4),
            "dud_high_lift": round(dud_stats["high"]["lift"], 4),
            "dud_high_count": dud_stats["high"]["count"],
        }

    return {
        "rows": int(len(df)),
        "by_prop": by_prop,
        "breakout_score": _average_nonnegative(breakout_scores),
        "dud_score": _average_nonnegative(dud_scores),
    }


def _prop_candidate_weights(base_weights: dict, prop_type: str, *,
                            offset: float | None = None,
                            variance: float | None = None,
                            blend: float | None = None,
                            shrinkage: float | None = None) -> dict:
    """Return a shallow-cloned weight config with a single prop adjusted."""
    weights = copy.deepcopy(base_weights)
    if offset is not None:
        weights.setdefault("prop_type_offsets", {})
        weights["prop_type_offsets"][prop_type] = round(float(offset), 4)
    if variance is not None:
        weights.setdefault("distribution_params", {})
        weights["distribution_params"].setdefault(prop_type, {})
        weights["distribution_params"][prop_type]["vr"] = round(float(variance), 4)
        weights.setdefault("variance_ratios", {})
        weights["variance_ratios"][prop_type] = round(float(variance), 4)
    if blend is not None:
        weights.setdefault("calibration_blend_weights", {})
        weights["calibration_blend_weights"][prop_type] = round(float(blend), 4)
    if shrinkage is not None:
        raw = weights.get("confidence_shrinkage", {"default": 0.70})
        if not isinstance(raw, dict):
            raw = {"default": float(raw)}
        raw[prop_type] = round(float(shrinkage), 4)
        weights["confidence_shrinkage"] = raw
    return weights


def optimize_model_parameters(train_df: pd.DataFrame, base_weights: dict) -> dict:
    """Sequentially tune offset, variance, blend, and shrinkage per prop."""
    tuned = copy.deepcopy(base_weights)
    recommendations = {}

    for prop_type, subset in train_df.groupby("prop_type"):
        if len(subset) < MODEL_MIN_TRAIN_ROWS:
            continue

        prop_recs = {}

        current_offset = float(tuned.get("prop_type_offsets", {}).get(prop_type, 0.0))
        best_offset = current_offset
        best_offset_metrics = evaluate_model_weights(subset, tuned)["by_prop"].get(prop_type, {})
        best_offset_score = (
            _metric_or_default(best_offset_metrics, "mae"),
            _metric_or_default(best_offset_metrics, "rmse"),
        )
        for offset in _offset_grid_for_prop(prop_type, subset, current_offset):
            candidate_weights = _prop_candidate_weights(tuned, prop_type, offset=offset)
            metrics = evaluate_model_weights(subset, candidate_weights)["by_prop"].get(prop_type, {})
            score = (
                _metric_or_default(metrics, "mae"),
                _metric_or_default(metrics, "rmse"),
            )
            if score < best_offset_score:
                best_offset = offset
                best_offset_score = score
                best_offset_metrics = metrics
        tuned = _prop_candidate_weights(tuned, prop_type, offset=best_offset)
        prop_recs["offset"] = {
            "from": round(current_offset, 4),
            "to": round(best_offset, 4),
            "train_mae": round(best_offset_metrics.get("mae", 0.0), 4),
            "train_bias": round(best_offset_metrics.get("bias", 0.0), 4),
        }

        current_variance = _active_variance_value(tuned, prop_type)
        cfg = get_distribution_config(prop_type, tuned)
        if cfg["dist_type"] != "betabinom":
            best_variance = current_variance
            best_variance_metrics = evaluate_model_weights(subset, tuned)["by_prop"].get(prop_type, {})
            best_variance_score = (
                _metric_or_default(best_variance_metrics, "log_loss"),
                _metric_or_default(best_variance_metrics, "brier_score"),
            )
            var_grid = [round(current_variance * mult, 4) for mult in VARIANCE_MULTIPLIERS]
            for variance in sorted(set(var_grid + [current_variance])):
                if variance <= 1.01:
                    continue
                candidate_weights = _prop_candidate_weights(tuned, prop_type, variance=variance)
                metrics = evaluate_model_weights(subset, candidate_weights)["by_prop"].get(prop_type, {})
                score = (
                    _metric_or_default(metrics, "log_loss"),
                    _metric_or_default(metrics, "brier_score"),
                )
                if score < best_variance_score:
                    best_variance = variance
                    best_variance_score = score
                    best_variance_metrics = metrics
            tuned = _prop_candidate_weights(tuned, prop_type, variance=best_variance)
            prop_recs["variance"] = {
                "from": round(current_variance, 4),
                "to": round(best_variance, 4),
                "train_log_loss": round(best_variance_metrics.get("log_loss", 0.0), 4),
                "train_brier": round(best_variance_metrics.get("brier_score", 0.0), 4),
            }

        current_blend = _calibration_blend_value(tuned, prop_type)
        if _empirical_probs_vectorized(np.array([subset["projection"].mean()]), prop_type, float(subset["line"].iloc[0])) is not None:
            best_blend = current_blend
            best_blend_metrics = evaluate_model_weights(subset, tuned)["by_prop"].get(prop_type, {})
            best_blend_score = (
                _metric_or_default(best_blend_metrics, "log_loss"),
                _metric_or_default(best_blend_metrics, "brier_score"),
            )
            for blend in sorted(set(BLEND_GRID + [current_blend])):
                candidate_weights = _prop_candidate_weights(tuned, prop_type, blend=blend)
                metrics = evaluate_model_weights(subset, candidate_weights)["by_prop"].get(prop_type, {})
                score = (
                    _metric_or_default(metrics, "log_loss"),
                    _metric_or_default(metrics, "brier_score"),
                )
                if score < best_blend_score:
                    best_blend = blend
                    best_blend_score = score
                    best_blend_metrics = metrics
            tuned = _prop_candidate_weights(tuned, prop_type, blend=best_blend)
            prop_recs["calibration_blend"] = {
                "from": round(current_blend, 4),
                "to": round(best_blend, 4),
                "train_log_loss": round(best_blend_metrics.get("log_loss", 0.0), 4),
            }

        current_shrink = _confidence_shrinkage_value(tuned, prop_type)
        best_shrink = current_shrink
        best_shrink_metrics = evaluate_model_weights(subset, tuned)["by_prop"].get(prop_type, {})
        best_shrink_score = (
            _metric_or_default(best_shrink_metrics, "log_loss"),
            _metric_or_default(best_shrink_metrics, "brier_score"),
        )
        for shrinkage in sorted(set(SHRINKAGE_GRID + [current_shrink])):
            candidate_weights = _prop_candidate_weights(tuned, prop_type, shrinkage=shrinkage)
            metrics = evaluate_model_weights(subset, candidate_weights)["by_prop"].get(prop_type, {})
            score = (
                _metric_or_default(metrics, "log_loss"),
                _metric_or_default(metrics, "brier_score"),
            )
            if score < best_shrink_score:
                best_shrink = shrinkage
                best_shrink_score = score
                best_shrink_metrics = metrics
        tuned = _prop_candidate_weights(tuned, prop_type, shrinkage=best_shrink)
        prop_recs["confidence_shrinkage"] = {
            "from": round(current_shrink, 4),
            "to": round(best_shrink, 4),
            "train_log_loss": round(best_shrink_metrics.get("log_loss", 0.0), 4),
        }

        recommendations[prop_type] = prop_recs

    return {"weights": tuned, "recommendations": recommendations}


def _wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Wilson lower confidence bound for a binomial success rate."""
    if total <= 0:
        return 0.0
    phat = wins / total
    denom = 1 + (z ** 2) / total
    center = phat + (z ** 2) / (2 * total)
    margin = z * ((phat * (1 - phat) + (z ** 2) / (4 * total)) / total) ** 0.5
    return (center - margin) / denom


def evaluate_floors(df: pd.DataFrame, floors: dict) -> dict:
    """Evaluate a floor configuration on a graded DataFrame."""
    if df.empty:
        return {
            "selected": 0,
            "available": 0,
            "accuracy": None,
            "brier_score": None,
            "log_loss": None,
            "coverage_pct": 0.0,
            "by_prop_direction": {},
        }

    work = df.copy()
    work["floor"] = work["floor_key"].map(lambda key: float(floors.get(key, 0.60)))
    selected = work[work["confidence"] >= work["floor"]].copy()

    result = {
        "selected": int(len(selected)),
        "available": int(len(work)),
        "coverage_pct": round(len(selected) / len(work), 4) if len(work) else 0.0,
        "accuracy": None,
        "brier_score": None,
        "log_loss": None,
        "by_prop_direction": {},
    }
    if selected.empty:
        return result

    result["accuracy"] = float(selected["is_win"].mean())
    probs = np.clip(selected["confidence"].astype(float).values, 1e-7, 1 - 1e-7)
    outcomes = selected["is_win"].astype(float).values
    result["brier_score"] = float(np.mean((probs - outcomes) ** 2))
    result["log_loss"] = float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))

    grouped = selected.groupby(["prop_type", "pick"])
    for (prop_type, direction), subset in grouped:
        result["by_prop_direction"][f"{prop_type}_{direction.lower()}"] = {
            "count": int(len(subset)),
            "accuracy": float(subset["is_win"].mean()),
        }

    return result


def optimize_confidence_floors(train_df: pd.DataFrame, base_floors: dict,
                               grid: list[float] | None = None) -> dict:
    """Optimize per-prop confidence floors on the training split."""
    grid = sorted(set((grid or DEFAULT_GRID) + [0.60]))
    tuned = copy.deepcopy(base_floors)
    recommendations = {}

    for floor_name, subset in train_df.groupby("floor_key"):
        if len(subset) < MIN_TRAIN_PICKS:
            continue

        current_floor = float(base_floors.get(floor_name, 0.60))
        candidates = sorted(set(grid + [current_floor]))
        best = None
        for floor in candidates:
            selected = subset[subset["confidence"] >= floor]
            if len(selected) < MIN_TRAIN_PICKS:
                continue
            wins = int(selected["is_win"].sum())
            total = int(len(selected))
            candidate = FloorCandidate(
                floor=floor,
                count=total,
                accuracy=float(selected["is_win"].mean()),
                score=_wilson_lower_bound(wins, total),
            )
            if best is None or candidate.score > best.score + 1e-9 or (
                abs(candidate.score - best.score) <= 1e-9 and candidate.count > best.count
            ):
                best = candidate

        if best is None:
            continue

        tuned[floor_name] = round(best.floor, 2)
        recommendations[floor_name] = {
            "from": round(current_floor, 2),
            "to": round(best.floor, 2),
            "train_count": best.count,
            "train_accuracy": round(best.accuracy, 4),
            "train_score": round(best.score, 4),
        }

    return {"floors": tuned, "recommendations": recommendations}


def analyze_backtest_floors(filepath: str = DEFAULT_RESULTS_PATH) -> dict:
    """Run the full offline floor-tuning analysis."""
    df = load_backtest_dataframe(filepath)
    if df.empty:
        return {"error": "No supported graded backtest results found."}

    current_weights = load_current_weights()
    current_floors = current_weights.get("per_prop_confidence_floors", {})

    train_df, valid_df, holdout_df = split_backtest_dataframe_three_way(df)
    if valid_df.empty:
        return {"error": "Not enough distinct backtest dates to build a validation split."}

    tuned = optimize_confidence_floors(train_df, current_floors)
    current_eval = evaluate_floors(valid_df, current_floors)
    candidate_eval = evaluate_floors(valid_df, tuned["floors"])
    holdout_current = evaluate_floors(holdout_df, current_floors) if not holdout_df.empty else {}
    holdout_candidate = evaluate_floors(holdout_df, tuned["floors"]) if not holdout_df.empty else {}

    current_selected = current_eval.get("selected", 0)
    candidate_selected = candidate_eval.get("selected", 0)
    current_accuracy = current_eval.get("accuracy") or 0.0
    candidate_accuracy = candidate_eval.get("accuracy") or 0.0
    holdout_accuracy_gain = (holdout_candidate.get("accuracy") or 0.0) - (holdout_current.get("accuracy") or 0.0)

    keep_volume = candidate_selected >= max(MIN_VALID_PICKS, int(current_selected * MIN_VOLUME_RETAIN))
    accuracy_gain = candidate_accuracy - current_accuracy
    should_apply = (
        candidate_selected >= MIN_VALID_PICKS
        and keep_volume
        and accuracy_gain >= MIN_ACCURACY_GAIN
        and (
            holdout_df.empty
            or (
                holdout_candidate.get("selected", 0) >= MIN_VALID_PICKS
                and holdout_accuracy_gain >= 0.0
            )
        )
    )

    return {
        "backtest_path": str(filepath),
        "train_dates": int(train_df["game_date"].dt.date.nunique()),
        "validation_dates": int(valid_df["game_date"].dt.date.nunique()),
        "holdout_dates": int(holdout_df["game_date"].dt.date.nunique()) if not holdout_df.empty else 0,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "holdout_rows": int(len(holdout_df)),
        "current": current_eval,
        "candidate": candidate_eval,
        "holdout_current": holdout_current,
        "holdout_candidate": holdout_candidate,
        "recommendations": tuned["recommendations"],
        "candidate_floors": tuned["floors"],
        "should_apply": should_apply,
        "reason": (
            f"validation accuracy {candidate_accuracy:.3f} vs {current_accuracy:.3f}, "
            f"selected {candidate_selected} vs {current_selected}, "
            f"holdout delta {holdout_accuracy_gain:.3f}"
        ),
    }


def apply_candidate_floors(analysis: dict) -> dict:
    """Promote tuned floors into a new weights version when warranted."""
    if analysis.get("error"):
        return {"applied": False, "reason": analysis["error"]}
    if not analysis.get("should_apply"):
        return {"applied": False, "reason": analysis.get("reason", "Candidate did not clear promotion gates.")}

    weights = load_current_weights()
    weights["per_prop_confidence_floors"] = copy.deepcopy(analysis["candidate_floors"])
    weights.setdefault("metadata", {})
    weights["metadata"]["offline_tuning"] = {
        "source_backtest": analysis.get("backtest_path"),
        "train_rows": analysis.get("train_rows"),
        "validation_rows": analysis.get("validation_rows"),
        "applied_at": pd.Timestamp.utcnow().isoformat(),
        "reason": analysis.get("reason"),
        "recommendations": analysis.get("recommendations", {}),
    }

    version = _next_version()
    description = "offline floor tuning from historical backtest"
    save_path = save_weights(weights, version, description)
    return {"applied": True, "version": version, "path": save_path, "reason": analysis.get("reason")}


def analyze_backtest_model(filepath: str = DEFAULT_RESULTS_PATH) -> dict:
    """Run the offline model-parameter tuning analysis."""
    df = load_model_backtest_dataframe(filepath)
    if df.empty:
        return {"error": "No graded model-tuning rows found in the backtest file."}

    train_df, valid_df, holdout_df = split_backtest_dataframe_three_way(df)
    if valid_df.empty:
        return {"error": "Not enough distinct backtest dates to build a validation split."}

    current_weights = load_current_weights()
    tuned = optimize_model_parameters(train_df, current_weights)

    current_eval = evaluate_model_weights(valid_df, current_weights)
    candidate_eval = evaluate_model_weights(valid_df, tuned["weights"])
    holdout_current = evaluate_model_weights(holdout_df, current_weights) if not holdout_df.empty else {}
    holdout_candidate = evaluate_model_weights(holdout_df, tuned["weights"]) if not holdout_df.empty else {}

    current_log_loss = _metric_or_default(current_eval, "log_loss")
    candidate_log_loss = _metric_or_default(candidate_eval, "log_loss")
    current_mae = _metric_or_default(current_eval, "mae")
    candidate_mae = _metric_or_default(candidate_eval, "mae")
    holdout_log_loss_gain = _metric_or_default(holdout_current, "log_loss") - _metric_or_default(holdout_candidate, "log_loss")
    holdout_mae_delta = _metric_or_default(holdout_candidate, "mae") - _metric_or_default(holdout_current, "mae")

    should_apply = (
        candidate_eval.get("rows", 0) >= MODEL_MIN_VALID_ROWS
        and (current_log_loss - candidate_log_loss) >= MODEL_MIN_LOGLOSS_GAIN
        and (candidate_mae - current_mae) <= MODEL_MAX_MAE_REGRESSION
        and (
            holdout_df.empty
            or (
                holdout_candidate.get("rows", 0) >= MODEL_MIN_VALID_ROWS
                and holdout_log_loss_gain >= 0.0
                and holdout_mae_delta <= MODEL_MAX_MAE_REGRESSION
            )
        )
    )

    return {
        "backtest_path": str(filepath),
        "train_dates": int(train_df["game_date"].dt.date.nunique()),
        "validation_dates": int(valid_df["game_date"].dt.date.nunique()),
        "holdout_dates": int(holdout_df["game_date"].dt.date.nunique()) if not holdout_df.empty else 0,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "holdout_rows": int(len(holdout_df)),
        "current": current_eval,
        "candidate": candidate_eval,
        "holdout_current": holdout_current,
        "holdout_candidate": holdout_candidate,
        "recommendations": tuned["recommendations"],
        "candidate_weights": {
            "prop_type_offsets": tuned["weights"].get("prop_type_offsets", {}),
            "distribution_params": tuned["weights"].get("distribution_params", {}),
            "variance_ratios": tuned["weights"].get("variance_ratios", {}),
            "calibration_blend_weights": tuned["weights"].get("calibration_blend_weights", {}),
            "confidence_shrinkage": tuned["weights"].get("confidence_shrinkage", {}),
        },
        "should_apply": should_apply,
        "reason": (
            f"validation log loss {candidate_log_loss:.4f} vs {current_log_loss:.4f}, "
            f"mae {candidate_mae:.4f} vs {current_mae:.4f}, "
            f"holdout log-loss delta {holdout_log_loss_gain:.4f}"
        ),
    }


def analyze_backtest_tail_signals(filepath: str = DEFAULT_RESULTS_PATH) -> dict:
    """Run offline tail-label threshold tuning on the historical backtest."""
    df = load_tail_backtest_dataframe(filepath)
    if df.empty:
        return {"error": "No tail-signal backtest rows found."}

    train_df, valid_df, holdout_df = split_backtest_dataframe_three_way(df)
    if valid_df.empty:
        return {"error": "Not enough distinct backtest dates to build a validation split."}

    current_weights = load_current_weights()
    current_cfg = current_weights.get("tail_signal_config", {})
    tuned = optimize_tail_signal_config(train_df, current_weights)

    current_eval = evaluate_tail_signal_config(valid_df, current_cfg)
    candidate_eval = evaluate_tail_signal_config(valid_df, tuned["tail_signal_config"])
    holdout_current = evaluate_tail_signal_config(holdout_df, current_cfg) if not holdout_df.empty else {}
    holdout_candidate = evaluate_tail_signal_config(holdout_df, tuned["tail_signal_config"]) if not holdout_df.empty else {}

    current_score = current_eval.get("breakout_score", 0.0) + current_eval.get("dud_score", 0.0)
    candidate_score = candidate_eval.get("breakout_score", 0.0) + candidate_eval.get("dud_score", 0.0)
    holdout_delta = (
        holdout_candidate.get("breakout_score", 0.0) + holdout_candidate.get("dud_score", 0.0)
        - holdout_current.get("breakout_score", 0.0) - holdout_current.get("dud_score", 0.0)
    )
    config_changed = _tail_configs_differ(current_cfg, tuned["tail_signal_config"])

    should_apply = (
        config_changed
        and (candidate_score - current_score) >= TAIL_MIN_SCORE_GAIN
        and (holdout_df.empty or holdout_delta >= TAIL_MIN_HOLDOUT_GAIN)
    )

    return {
        "backtest_path": str(filepath),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "holdout_rows": int(len(holdout_df)),
        "current": current_eval,
        "candidate": candidate_eval,
        "holdout_current": holdout_current,
        "holdout_candidate": holdout_candidate,
        "candidate_tail_signal_config": tuned["tail_signal_config"],
        "recommendations": tuned["recommendations"],
        "should_apply": should_apply,
        "reason": (
            f"validation tail score {candidate_score:.4f} vs {current_score:.4f}, "
            f"holdout delta {holdout_delta:.4f}, config_changed={config_changed}"
        ),
    }


def apply_candidate_tail_signals(analysis: dict) -> dict:
    """Promote tuned tail-label thresholds into a new weights version when warranted."""
    if analysis.get("error"):
        return {"applied": False, "reason": analysis["error"]}
    if not analysis.get("should_apply"):
        return {"applied": False, "reason": analysis.get("reason", "Tail candidate did not clear promotion gates.")}

    weights = load_current_weights()
    weights["tail_signal_config"] = copy.deepcopy(analysis.get("candidate_tail_signal_config", {}))
    weights.setdefault("metadata", {})
    weights["metadata"]["offline_tail_tuning"] = {
        "source_backtest": analysis.get("backtest_path"),
        "train_rows": analysis.get("train_rows"),
        "validation_rows": analysis.get("validation_rows"),
        "applied_at": pd.Timestamp.utcnow().isoformat(),
        "reason": analysis.get("reason"),
        "recommendations": analysis.get("recommendations", {}),
    }

    version = _next_version()
    description = "offline tail threshold tuning from historical backtest"
    save_path = save_weights(weights, version, description)
    return {"applied": True, "version": version, "path": save_path, "reason": analysis.get("reason")}


def apply_candidate_model(analysis: dict) -> dict:
    """Promote tuned model parameters into a new weights version when warranted."""
    if analysis.get("error"):
        return {"applied": False, "reason": analysis["error"]}
    if not analysis.get("should_apply"):
        return {"applied": False, "reason": analysis.get("reason", "Candidate model did not clear promotion gates.")}

    weights = load_current_weights()
    candidate = analysis.get("candidate_weights", {})
    for key in (
        "prop_type_offsets",
        "distribution_params",
        "variance_ratios",
        "calibration_blend_weights",
        "confidence_shrinkage",
    ):
        if key in candidate:
            weights[key] = copy.deepcopy(candidate[key])

    weights.setdefault("metadata", {})
    weights["metadata"]["offline_model_tuning"] = {
        "source_backtest": analysis.get("backtest_path"),
        "train_rows": analysis.get("train_rows"),
        "validation_rows": analysis.get("validation_rows"),
        "applied_at": pd.Timestamp.utcnow().isoformat(),
        "reason": analysis.get("reason"),
        "recommendations": analysis.get("recommendations", {}),
    }

    version = _next_version()
    description = "offline model tuning from historical backtest"
    save_path = save_weights(weights, version, description)
    return {"applied": True, "version": version, "path": save_path, "reason": analysis.get("reason")}


def optimize_floors(backtest_path: str = DEFAULT_RESULTS_PATH) -> dict:
    """
    Compatibility wrapper for automation scripts.

    Returns a lightweight status payload without mutating tracked weights.
    """
    analysis = analyze_backtest_floors(backtest_path)
    if analysis.get("error"):
        return {"status": "error", "reason": analysis["error"], "changes": {}}
    if not analysis.get("should_apply"):
        return {
            "status": "no_change",
            "reason": analysis.get("reason"),
            "changes": {},
            "analysis": analysis,
        }
    current = load_current_weights().get("per_prop_confidence_floors", {})
    candidate = analysis.get("candidate_floors", {})
    changes = {
        key: value
        for key, value in candidate.items()
        if round(float(current.get(key, 0.0)), 4) != round(float(value), 4)
    }
    return {
        "status": "updated" if changes else "no_change",
        "reason": analysis.get("reason"),
        "changes": changes,
        "analysis": analysis,
    }


def tune_model_parameters(backtest_path: str = DEFAULT_RESULTS_PATH) -> dict:
    """
    Compatibility wrapper for automation scripts.

    Returns only the candidate overrides that should be written to runtime
    weights if holdout gates pass.
    """
    analysis = analyze_backtest_model(backtest_path)
    if analysis.get("error"):
        return {"status": "error", "reason": analysis["error"], "changes": {}}
    if not analysis.get("should_apply"):
        return {
            "status": "no_change",
            "reason": analysis.get("reason"),
            "changes": {},
            "analysis": analysis,
        }
    return {
        "status": "updated",
        "reason": analysis.get("reason"),
        "changes": analysis.get("candidate_weights", {}),
        "analysis": analysis,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline confidence-floor tuner")
    parser.add_argument("--backtest-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--model", action="store_true", help="Tune model offsets/variance/calibration instead of selection floors.")
    parser.add_argument("--tail", action="store_true", help="Tune breakout/dud label thresholds from the backtest.")
    parser.add_argument("--rebuild-calibration", action="store_true", help="Rebuild empirical calibration tables from the backtest.")
    parser.add_argument("--apply-if-better", action="store_true")
    args = parser.parse_args(argv)

    if args.rebuild_calibration:
        analysis = rebuild_backtest_calibration_tables(args.backtest_path)
        print(json.dumps(analysis, indent=2, default=str))
        return 0

    if args.tail:
        analysis = analyze_backtest_tail_signals(args.backtest_path)
    elif args.model:
        analysis = analyze_backtest_model(args.backtest_path)
    else:
        analysis = analyze_backtest_floors(args.backtest_path)
    print(json.dumps(analysis, indent=2, default=str))

    if args.apply_if_better:
        if args.tail:
            result = apply_candidate_tail_signals(analysis)
        elif args.model:
            result = apply_candidate_model(analysis)
        else:
            result = apply_candidate_floors(analysis)
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
