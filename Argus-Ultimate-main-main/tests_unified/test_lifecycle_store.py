from argus_live.promotion.lifecycle_store import LifecycleStore

def test_lifecycle_store_roundtrip(tmp_path) -> None:
    store = LifecycleStore(tmp_path / "strategy_lifecycle.json")
    store.save({"s1": "LIVE_SMALL"})
    loaded = store.load()
    assert loaded["s1"] == "LIVE_SMALL"
