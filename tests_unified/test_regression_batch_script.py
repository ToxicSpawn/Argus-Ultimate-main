from pathlib import Path
import subprocess
import sys


def test_regression_batch_script_runs(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts/run_regression_batch.py"), "--artifacts-root", str(tmp_path), "--batch-name", "test-batch"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode in (0, 1)
    assert '"batch_name": "test-batch"' in proc.stdout
    assert (tmp_path / 'test-batch' / 'batch_summary.json').exists()
