import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kelly import PAYOUT_MULTIPLIERS
from src.slips import PAYOUTS


class PayoutTableTests(unittest.TestCase):
    def test_kelly_uses_same_perfect_hit_multiplier_as_slip_tables(self):
        for entry_type, payout_table in PAYOUTS.items():
            expected = max((float(mult) for mult in payout_table.values()), default=1.0)
            self.assertEqual(PAYOUT_MULTIPLIERS[entry_type], expected)

    def test_standard_tables_match_current_minimum_guarantee_values(self):
        self.assertEqual(PAYOUTS["3_power"][3], 5.0)
        self.assertEqual(PAYOUTS["3_flex"][3], 2.25)
        self.assertEqual(PAYOUTS["3_flex"][2], 1.25)
        self.assertEqual(PAYOUTS["4_flex"][4], 5.0)


if __name__ == "__main__":
    unittest.main()
