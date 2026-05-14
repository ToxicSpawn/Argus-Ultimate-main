from argus_live.promotion.autonomous_promotion import evaluate_for_promotion
from argus_live.promotion.promotion_bundle import PromotionBundle

def test_autonomous_promotion_accepts_good_bundle() -> None:
    bundle = PromotionBundle(
        strategy_id="s1",
        feature_hash="abc",
        training_window="2025",
        evaluation_window="2026",
        walk_forward_score=1.2,
        stress_score=1.1,
        replay_passed=True,
        approved_by="operator",
        approved_at_utc="2026-01-01T00:00:00Z",
        signature="sig",
    )
    decision = evaluate_for_promotion(bundle)
    assert decision.promote is True
