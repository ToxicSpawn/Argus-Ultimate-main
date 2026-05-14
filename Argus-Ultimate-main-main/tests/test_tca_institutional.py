import csv
import os
import sys
import tempfile
import unittest
from datetime import datetime


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from monitoring.tca_institutional import (  # noqa: E402
    ExecutionQualityScorer,
    FillQuality,
    MarketImpactModel,
    TCADashboard,
    VenueAnalyzer,
)


class TestInstitutionalTCA(unittest.TestCase):
    def _fills(self):
        return [
            FillQuality(
                fill_id="f1",
                venue="alpha",
                timestamp=datetime(2026, 4, 22, 9, 30),
                side="buy",
                quantity=5.0,
                price=100.2,
                benchmark_price=100.1,
                arrival_price=100.0,
                implementation_shortfall=4.0,
                market_impact=1.5,
                timing_cost=0.5,
            ),
            FillQuality(
                fill_id="f2",
                venue="alpha",
                timestamp=datetime(2026, 4, 22, 10, 0),
                side="buy",
                quantity=4.0,
                price=100.1,
                benchmark_price=100.05,
                arrival_price=100.0,
                implementation_shortfall=2.5,
                market_impact=1.0,
                timing_cost=0.5,
            ),
            FillQuality(
                fill_id="f3",
                venue="beta",
                timestamp=datetime(2026, 4, 22, 10, 15),
                side="buy",
                quantity=5.0,
                price=100.5,
                benchmark_price=100.4,
                arrival_price=100.0,
                implementation_shortfall=8.0,
                market_impact=3.0,
                timing_cost=1.0,
            ),
        ]

    def test_venue_analyzer_comparison_and_rates(self):
        analyzer = VenueAnalyzer(
            fills=self._fills(),
            venue_order_stats={
                "alpha": {"attempted": 3, "filled": 2, "rejected": 1},
                "beta": {"attempted": 2, "filled": 1, "rejected": 1},
            },
        )

        alpha_metrics = analyzer.get_venue_metrics("alpha")
        comparison = analyzer.compare_venues()

        self.assertEqual(alpha_metrics["fill_count"], 2)
        self.assertAlmostEqual(analyzer.compute_fill_rate("alpha"), 66.66666666666666)
        self.assertAlmostEqual(analyzer.compute_reject_rate("beta"), 50.0)
        self.assertEqual(comparison["best_venue"], "alpha")
        self.assertEqual(analyzer.get_best_venue("buy", 5.0), "alpha")

    def test_market_impact_model_and_shortfall(self):
        model = MarketImpactModel()
        params = model.fit_model([
            {"quantity": 10_000, "market_impact": 6.0, "adv": 1_000_000},
            {"quantity": 40_000, "market_impact": 11.0, "adv": 1_000_000},
            {"quantity": 90_000, "market_impact": 17.0, "adv": 1_000_000},
        ])

        prediction = model.predict_impact("buy", 50_000, 1_000_000)
        shortfall = model.compute_implementation_shortfall(
            {"side": "buy", "quantity": 12.0, "arrival_price": 100.0},
            self._fills()[:2],
        )

        self.assertIn("eta", params)
        self.assertGreater(prediction, 0.0)
        self.assertGreater(shortfall["implementation_shortfall_bps"], 0.0)
        self.assertGreater(shortfall["opportunity_cost_bps"], 0.0)

    def test_dashboard_export_and_scoring(self):
        fills = self._fills()
        dashboard = TCADashboard(
            fills=fills,
            venue_order_stats={
                "alpha": {"attempted": 3, "filled": 2, "rejected": 1},
                "beta": {"attempted": 2, "filled": 1, "rejected": 1},
            },
        )

        report = dashboard.generate_daily_report()
        venue_report = dashboard.generate_venue_comparison()
        attribution = dashboard.generate_cost_attribution()
        scorer = ExecutionQualityScorer(dashboard.venue_analyzer)

        self.assertEqual(report["summary"]["fill_count"], 3)
        self.assertEqual(venue_report["best_venue"], "alpha")
        self.assertGreater(attribution["totals"]["total_cost"], 0.0)
        self.assertGreaterEqual(scorer.score_fill(fills[0]), 0.0)
        self.assertLessEqual(scorer.score_venue("alpha"), 100.0)
        self.assertTrue(scorer.get_improvement_suggestions())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "tca_export.csv")
            exported = dashboard.export_to_csv(output_path)
            self.assertEqual(exported, output_path)

            with open(output_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["fill_id"], "f1")


if __name__ == "__main__":
    unittest.main()
