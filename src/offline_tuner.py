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

from src.autolearn import load_current_weights, save_weights, _next_version
from src.backtester import DEFAULT_RESULTS_PATH, load_results, filter_nonplays
from src.selection import floor_key, get_confidence_floor


SUPPORTED_PROPS = {
    "hits",
    "total_bases",
    "pitcher_strikeouts",
    "hitter_fantasy_score",
}
DEFAULT_GRID = [round(x, 2) for x in np.arange(0.54, 0.91, 0.02)]
MIN_TRAIN_PICKS = 80
MIN_VALID_PICKS = 30
MIN_VOLUME_RETAIN = 0.85
MIN_ACCURACY_GAIN = 0.003


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

    train_df, valid_df = split_backtest_dataframe(df)
    if valid_df.empty:
        return {"error": "Not enough distinct backtest dates to build a validation split."}

    tuned = optimize_confidence_floors(train_df, current_floors)
    current_eval = evaluate_floors(valid_df, current_floors)
    candidate_eval = evaluate_floors(valid_df, tuned["floors"])

    current_selected = current_eval.get("selected", 0)
    candidate_selected = candidate_eval.get("selected", 0)
    current_accuracy = current_eval.get("accuracy") or 0.0
    candidate_accuracy = candidate_eval.get("accuracy") or 0.0

    keep_volume = candidate_selected >= max(MIN_VALID_PICKS, int(current_selected * MIN_VOLUME_RETAIN))
    accuracy_gain = candidate_accuracy - current_accuracy
    should_apply = (
        candidate_selected >= MIN_VALID_PICKS
        and keep_volume
        and accuracy_gain >= MIN_ACCURACY_GAIN
    )

    return {
        "backtest_path": str(filepath),
        "train_dates": int(train_df["game_date"].dt.date.nunique()),
        "validation_dates": int(valid_df["game_date"].dt.date.nunique()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "current": current_eval,
        "candidate": candidate_eval,
        "recommendations": tuned["recommendations"],
        "candidate_floors": tuned["floors"],
        "should_apply": should_apply,
        "reason": (
            f"validation accuracy {candidate_accuracy:.3f} vs {current_accuracy:.3f}, "
            f"selected {candidate_selected} vs {current_selected}"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline confidence-floor tuner")
    parser.add_argument("--backtest-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--apply-if-better", action="store_true")
    args = parser.parse_args(argv)

    analysis = analyze_backtest_floors(args.backtest_path)
    print(json.dumps(analysis, indent=2, default=str))

    if args.apply_if_better:
        result = apply_candidate_floors(analysis)
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
