import json
from pathlib import Path

from argus_live.simulation.batch_runner import ScenarioBatchRunner
from argus_live.simulation.regression_library import build_regression_library


def test_scenario_batch_runner_writes_summary(tmp_path: Path):
    runner = ScenarioBatchRunner(tmp_path)
    scenarios = build_regression_library()[:3]
    result = runner.run(batch_name="regression", scenarios=scenarios)
    assert result.overall_score >= 0.0
    assert result.pass_rate >= 0.0
    assert len(result.scenario_results) == 3
    payload = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert payload["batch_name"] == "regression"
    assert len(payload["scenario_results"]) == 3
