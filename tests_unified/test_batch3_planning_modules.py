from execution.algo_orders import build_twap_plan, build_vwap_style_plan
from execution.delta_neutral_executor import DeltaNeutralExecutor
from execution.pipeline import ExecutionPipeline, ExecutionPipelineInput
from core.execution.advanced_orders import build_post_only_template, build_reduce_only_template
from core.execution.order_with_stop import build_order_with_stop_plan


def test_execution_pipeline_returns_plan_only() -> None:
    pipeline = ExecutionPipeline()
    plan = pipeline.build_plan(
        ExecutionPipelineInput(
            symbol="BTC/AUD",
            quantity=1.0,
            reference_price=50_000.0,
            top_of_book_notional=100_000.0,
            spread_bps=3.0,
            volatility_bps=10.0,
            allow_market_orders=False,
        )
    )
    assert plan.liquidity.approved_quantity > 0
    assert plan.route.venue == "kraken"
    assert plan.slippage.expected_bps >= 0


def test_algo_order_helpers_build_schedules() -> None:
    twap = build_twap_plan(total_quantity=10.0, duration_seconds=600, slice_count=5)
    assert twap.slice_quantity == 2.0
    vwap = build_vwap_style_plan(total_quantity=10.0, expected_participation=0.25)
    assert vwap.slice_count >= 1


def test_delta_neutral_executor_is_advisory_only() -> None:
    executor = DeltaNeutralExecutor()
    suggestion = executor.suggest_hedge(primary_symbol="BTC/AUD", hedge_symbol="ETH/AUD", primary_notional=10_000.0, hedge_ratio=0.5)
    assert suggestion.hedge_notional == 5_000.0


def test_advanced_order_templates_are_plans_only() -> None:
    maker = build_post_only_template()
    assert maker.post_only is True
    reducer = build_reduce_only_template()
    assert reducer.reduce_only is True


def test_order_with_stop_plan_builds_expected_levels() -> None:
    plan = build_order_with_stop_plan(entry_price=100.0, stop_distance_pct=0.05, take_profit_distance_pct=0.10)
    assert plan.stop_price == 95.0
    assert plan.take_profit_price == 110.0
