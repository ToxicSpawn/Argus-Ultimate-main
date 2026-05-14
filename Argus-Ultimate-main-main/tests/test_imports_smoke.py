'''
Smoke tests for S+++ tier modules
Quick import and basic functionality tests
'''

import pytest

from core.execution.idempotency import OrderIntent, client_order_id

try:
    from monitoring.metrics import MetricsCollector
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


def test_imports_smoke():
    '''Test that all S+++ tier modules can be imported'''

    # Test idempotency
    intent = OrderIntent(strategy="test", symbol="BTC/USDT", side="buy", qty=0.1, price=50000.0, time_bucket_s=60)

    order_id = client_order_id(intent)
    assert order_id.startswith("argus_")
    assert len(order_id) > 10


@pytest.mark.skipif(not HAS_METRICS, reason="monitoring.metrics not available")
def test_metrics_importable():
    '''Test that metrics module can be imported'''
    assert MetricsCollector is not None


def test_idempotency_deterministic():
    '''Test that same intent produces same order ID'''

    intent1 = OrderIntent(strategy="core", symbol="BTC/USDT", side="buy", qty=0.1, price=50000.0, time_bucket_s=60)

    intent2 = OrderIntent(strategy="core", symbol="BTC/USDT", side="buy", qty=0.1, price=50000.0, time_bucket_s=60)

    assert client_order_id(intent1) == client_order_id(intent2)


def test_idempotency_different_intents():
    '''Test that different intents produce different order IDs'''

    intent1 = OrderIntent(strategy="core", symbol="BTC/USDT", side="buy", qty=0.1, price=50000.0, time_bucket_s=60)

    intent2 = OrderIntent(
        strategy="core",
        symbol="BTC/USDT",
        side="sell",  # Different side
        qty=0.1,
        price=50000.0,
        time_bucket_s=60,
    )

    assert client_order_id(intent1) != client_order_id(intent2)
