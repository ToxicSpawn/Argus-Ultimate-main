from __future__ import annotations

import unittest

from argus_live.portfolio.meta_allocator import allocate_by_family


class TestMetaAllocator(unittest.TestCase):
    def test_family_allocator_weights_capital(self) -> None:
        allocs = allocate_by_family(
            total_capital=10_000.0,
            family_scores={"trend": 2.0, "mr": 1.0},
        )
        by_family = {a.family: a for a in allocs}
        self.assertGreater(by_family["trend"].capital, by_family["mr"].capital)
        self.assertAlmostEqual(
            by_family["trend"].capital + by_family["mr"].capital, 10_000.0, places=2
        )


if __name__ == "__main__":
    unittest.main()
