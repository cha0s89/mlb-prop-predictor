import unittest

from src.spring import apply_seasonal_spring_blend
from src.weather import _classify_field_relative_wind, get_stat_specific_weather_adjustment


class PreseasonSetupTests(unittest.TestCase):
    def test_spring_blend_is_full_strength_pre_opening_day(self):
        mult = apply_seasonal_spring_blend(
            1.08,
            game_date="2026-03-24",
            current_sample=0,
            is_pitcher=False,
            prop_type="total_bases",
        )
        self.assertAlmostEqual(mult, 1.08, places=3)

    def test_spring_blend_fades_with_real_sample(self):
        mult = apply_seasonal_spring_blend(
            1.08,
            game_date="2026-07-01",
            current_sample=250,
            is_pitcher=False,
            prop_type="total_bases",
        )
        self.assertLess(abs(mult - 1.0), 0.01)

    def test_field_relative_wind_distinguishes_out_vs_in(self):
        out_ctx = _classify_field_relative_wind(18, 247.5, 67.5)
        in_ctx = _classify_field_relative_wind(18, 67.5, 67.5)

        out_mult = get_stat_specific_weather_adjustment(
            {"temp_f": 72, "wind_mph": 18, "is_dome": False, **out_ctx},
            "home_runs",
        )
        in_mult = get_stat_specific_weather_adjustment(
            {"temp_f": 72, "wind_mph": 18, "is_dome": False, **in_ctx},
            "home_runs",
        )

        self.assertGreater(out_mult, 1.0)
        self.assertLess(in_mult, 1.0)


if __name__ == "__main__":
    unittest.main()
