from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from argus_live.proving.control_surface import build_control_surface
from argus_live.proving.day_review import DayReview
from argus_live.proving.operator_summary import write_operator_summary
from argus_live.proving.runner import run_proving_review


@dataclass
class ProvingLifecycleResult:
    run_id: str
    report_path: str
    review: DayReview
    regression_summary_path: str | None = None
    comparative_summary_path: str | None = None
    config_quarantine_path: str | None = None
    control_surface_path: str | None = None


class AutoProvingLifecycle:
    def __init__(
        self,
        *,
        incidents_path: str | Path,
        replay_audit_path: str | Path,
        report_path: str | Path,
        operator_summary_path: Optional[str | Path] = None,
        control_surface_path: Optional[str | Path] = None,
        regression_artifacts_root: Optional[str | Path] = None,
        enable_regression_suite: bool = False,
        baseline_report_path: Optional[str | Path] = None,
        baseline_config_path: Optional[str | Path] = None,
        config_quarantine_path: Optional[str | Path] = None,
    ) -> None:
        self.incidents_path = Path(incidents_path)
        self.replay_audit_path = Path(replay_audit_path)
        self.report_path = Path(report_path)
        self.operator_summary_path = Path(operator_summary_path) if operator_summary_path is not None else None
        self.control_surface_path = Path(control_surface_path) if control_surface_path is not None else self.report_path.parent / 'control_surface.json'
        self.regression_artifacts_root = Path(regression_artifacts_root) if regression_artifacts_root is not None else self.report_path.parent / 'regression'
        self.enable_regression_suite = enable_regression_suite
        self.baseline_report_path = Path(baseline_report_path) if baseline_report_path is not None else None
        self.baseline_config_path = Path(baseline_config_path) if baseline_config_path is not None else None
        self.config_quarantine_path = Path(config_quarantine_path) if config_quarantine_path is not None else self.report_path.parent / 'config_quarantine.jsonl'
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        if self.operator_summary_path is not None:
            self.operator_summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.control_surface_path.parent.mkdir(parents=True, exist_ok=True)

    def refresh(self, *, run_id: str, trade_ledger) -> ProvingLifecycleResult:
        regression_summary_path = ''
        if self.enable_regression_suite:
            from argus_live.simulation.batch_runner import ScenarioBatchRunner
            from argus_live.simulation.regression_library import build_regression_library

            batch = ScenarioBatchRunner(self.regression_artifacts_root).run(
                batch_name=f'{run_id}-regression',
                scenarios=build_regression_library(),
            )
            regression_summary_path = batch.summary_path
        review = run_proving_review(
            run_id=run_id,
            trade_ledger=trade_ledger,
            incidents_path=self.incidents_path,
            replay_audit_path=self.replay_audit_path,
            report_path=self.report_path,
            regression_summary_path=regression_summary_path or None,
            baseline_report_path=self.baseline_report_path,
            baseline_config_path=self.baseline_config_path,
            candidate_config_path=self.report_path.parent.parent / 'config' / f'{run_id}.json',
            config_quarantine_path=self.config_quarantine_path,
        )
        if self.operator_summary_path is not None:
            write_operator_summary(review, self.operator_summary_path)
        self.control_surface_path.write_text(__import__('json').dumps(build_control_surface(review), indent=2, sort_keys=True), encoding='utf-8')
        return ProvingLifecycleResult(
            run_id=run_id,
            report_path=str(self.report_path),
            review=review,
            regression_summary_path=regression_summary_path or None,
            comparative_summary_path=review.comparative_summary_path or None,
            config_quarantine_path=str(self.config_quarantine_path) if self.config_quarantine_path is not None else None,
            control_surface_path=str(self.control_surface_path),
        )
