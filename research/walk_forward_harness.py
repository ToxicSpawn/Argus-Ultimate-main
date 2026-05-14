from __future__ import annotations
import logging

import logging

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


logger = logging.getLogger(__name__)

@dataclass
class WalkForwardResult:
    report_path: str
    bundle_path: str
    metrics: Dict[str, Any]


def _git_commit() -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception as _e:
        logger.debug("walk_forward_harness error: %s", _e)
    return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _bundle_dir(ts: str, root: Path) -> Path:
    out = root / "deploy" / "bundles" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_bundle(
    *,
    ts: str,
    root: Path,
    config_path: Path,
    params: Dict[str, Any],
) -> Tuple[Path, List[Tuple[str, str]]]:
    bundle = _bundle_dir(ts, root)
    config_dst = bundle / "config.yaml"
    params_dst = bundle / "params.json"
    build_dst = bundle / "build_info.json"
    changelog_dst = bundle / "changelog.md"

    config_dst.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    params_dst.write_text(json.dumps(params, indent=2, ensure_ascii=True), encoding="utf-8")
    build_info = {
        "timestamp": ts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "bundle_version": "wf-skeleton-v1",
    }
    build_dst.write_text(json.dumps(build_info, indent=2, ensure_ascii=True), encoding="utf-8")
    changelog_dst.write_text(
        "\n".join(
            [
                "# Deploy Bundle Changelog",
                "",
                f"- bundle_timestamp: {ts}",
                "- source: walk_forward_harness_skeleton",
                "- notes: initial scaffold bundle with deterministic placeholder metrics",
            ]
        ),
        encoding="utf-8",
    )

    hashes = [
        ("config.yaml", _sha256(config_dst)),
        ("params.json", _sha256(params_dst)),
        ("build_info.json", _sha256(build_dst)),
        ("changelog.md", _sha256(changelog_dst)),
    ]
    hashes_dst = bundle / "hashes.txt"
    hashes_dst.write_text("\n".join([f"{h}  {n}" for n, h in hashes]), encoding="utf-8")
    return bundle, hashes


def run_walk_forward(
    *,
    config_path: str = "unified_config.yaml",
    report_dir: str = "reports",
    root_dir: str = ".",
) -> WalkForwardResult:
    root = Path(root_dir).resolve()
    cfg = Path(config_path).resolve()
    report_out = (root / report_dir).resolve()
    report_out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Skeleton metrics: deterministic placeholder for champion/challenger gate wiring.
    folds = []
    for i in range(1, 4):
        folds.append(
            {
                "fold": i,
                "train_start": f"window_{i}_train_start",
                "train_end": f"window_{i}_train_end",
                "test_start": f"window_{i}_test_start",
                "test_end": f"window_{i}_test_end",
                "sharpe": 0.0,
                "max_drawdown_pct": 0.0,
                "net_return_pct": 0.0,
            }
        )

    params = {
        "promotion_gate": "placeholder_champion_challenger",
        "notes": "Skeleton harness; plug in realistic backtest evaluator next.",
    }
    bundle, hashes = _build_bundle(ts=ts, root=root, config_path=cfg, params=params)

    report = {
        "timestamp": ts,
        "config": str(cfg),
        "folds": folds,
        "summary": {
            "avg_sharpe": 0.0,
            "avg_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "gate": "HOLD",
        },
        "bundle_path": str(bundle),
        "bundle_hashes": [{"file": n, "sha256": h} for n, h in hashes],
    }
    report_path = report_out / f"walk_forward_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return WalkForwardResult(
        report_path=str(report_path),
        bundle_path=str(bundle),
        metrics=report["summary"],
    )
