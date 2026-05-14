from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Iterable, List

from .hostile_replay import HarnessResult, HostileReplayHarness
from .regression_library import NamedScenario


@dataclass(frozen=True)
class ScenarioBatchScore:
    scenario_name: str
    promotion_decision: str
    score: float
    governance_response_score: float
    replay_score: float
    stability_score: float
    route_quality_score: float
    adverse_selection_damage_bps: float
    trade_count: int
    critical_incident_count: int
    replay_ok: bool
    active_blockers: list[str]
    report_path: str
    operator_summary_path: str


@dataclass(frozen=True)
class BatchRunResult:
    batch_name: str
    overall_score: float
    pass_rate: float
    scenario_results: List[ScenarioBatchScore]
    summary_path: str


_DECISION_BASE = {"GO": 100.0, "HOLD": 72.0, "NO_GO": 35.0}


def _load_ledger_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _component_scores(summary: dict, ledger_rows: list[dict]) -> tuple[float, float, float, float, float, float]:
    blockers = len(summary.get('active_blockers', []) or [])
    adverse_damage = round(sum(max(0.0, float(r.get('adverse_price_move_bps', 0.0) or 0.0)) for r in ledger_rows) / max(1, len(ledger_rows)), 2)
    governance = max(0.0, 100.0 - blockers * 12.5 - float(summary.get('critical_incident_count', 0) or 0) * 10.0)
    replay = 100.0 if bool(summary.get('replay_ok', False)) else 0.0
    stability = max(0.0, 100.0 - max(0.0, float(summary.get('max_drawdown_pct', 0.0) or 0.0)) * 12.0 - max(0.0, float(summary.get('reject_rate_pct', 0.0) or 0.0)) * 1.5)
    route_quality = max(0.0, 100.0 - max(0.0, float(summary.get('slippage_tail_bps', 0.0) or 0.0) - 10.0) * 1.3 - max(0.0, -float(summary.get('execution_alpha_bps', 0.0) or 0.0)) * 4.0)
    adverse_component = max(0.0, 100.0 - adverse_damage * 6.0)
    overall = round((governance * 0.20) + (replay * 0.20) + (stability * 0.20) + (route_quality * 0.25) + (adverse_component * 0.15), 2)
    return overall, round(governance, 2), round(replay, 2), round(stability, 2), round(route_quality, 2), adverse_damage


class ScenarioBatchRunner:
    def __init__(self, artifacts_root: str | Path, *, strict_governance: bool = True) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.strict_governance = strict_governance
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def run(self, *, batch_name: str, scenarios: Iterable[NamedScenario]) -> BatchRunResult:
        scenario_scores: List[ScenarioBatchScore] = []

        for named in scenarios:
            scenario_dir = self.artifacts_root / batch_name / named.name
            harness = HostileReplayHarness(
                scenario_dir,
                run_id=f"{batch_name}-{named.name}",
                strict_governance=self.strict_governance,
            )
            result: HarnessResult = harness.run(scenario=named.plan, orders=named.orders)
            summary = json.loads(Path(result.operator_summary_path).read_text(encoding="utf-8"))
            ledger_rows = _load_ledger_rows(result.ledger_path)
            score, governance_score, replay_score, stability_score, route_quality_score, adverse_damage = _component_scores(summary, ledger_rows)
            scenario_scores.append(
                ScenarioBatchScore(
                    scenario_name=named.name,
                    promotion_decision=str(summary.get("promotion_decision", "NO_GO")),
                    score=score,
                    governance_response_score=governance_score,
                    replay_score=replay_score,
                    stability_score=stability_score,
                    route_quality_score=route_quality_score,
                    adverse_selection_damage_bps=adverse_damage,
                    trade_count=int(summary.get("trade_count", 0) or 0),
                    critical_incident_count=int(summary.get("critical_incident_count", 0) or 0),
                    replay_ok=bool(summary.get("replay_ok", False)),
                    active_blockers=list(summary.get("active_blockers", []) or []),
                    report_path=result.report_path,
                    operator_summary_path=result.operator_summary_path,
                )
            )

        overall = round(mean([s.score for s in scenario_scores]), 2) if scenario_scores else 0.0
        pass_rate = 0.0 if not scenario_scores else round(100.0 * sum(1 for s in scenario_scores if s.promotion_decision != "NO_GO") / len(scenario_scores), 2)
        summary_path = self.artifacts_root / batch_name / "batch_summary.json"
        payload = {
            "batch_name": batch_name,
            "overall_score": overall,
            "pass_rate": pass_rate,
            "scenario_results": [asdict(s) for s in scenario_scores],
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return BatchRunResult(
            batch_name=batch_name,
            overall_score=overall,
            pass_rate=pass_rate,
            scenario_results=scenario_scores,
            summary_path=str(summary_path),
        )
