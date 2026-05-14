from argus_live.control_plane.runtime_manifest import build_manifest


def test_manifest_hashes_are_present() -> None:
    manifest = build_manifest(
        profile="live_safe",
        constitution_cfg={"constitution": {"version": 1}},
        runtime_cfg={"runtime": {"environment": "desktop_windows"}},
        exchange_cfg={"exchange": {"primary": "kraken"}},
        strategy_cfg={"approved_live": {"strategies": ["dca_kraken"]}},
        git_commit="abc123",
        node_role="execution-node",
    )
    assert manifest.manifest.constitution_hash.startswith("sha256:")
    assert manifest.manifest_hash.startswith("sha256:")
