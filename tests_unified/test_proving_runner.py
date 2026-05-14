import json
import tempfile

from argus_live.ledger.trade_ledger import LedgerFillRecord, TradeLedger
from argus_live.proving.runner import run_proving_review
from argus_live.replay.replay_audit import ReplayAudit, ReplayAuditStore


def test_proving_runner_generates_report() -> None:
    with tempfile.NamedTemporaryFile() as ledger_file, tempfile.NamedTemporaryFile() as incidents_file, tempfile.NamedTemporaryFile() as replay_file, tempfile.NamedTemporaryFile() as report_file:
        ledger = TradeLedger(ledger_file.name)
        ledger.append_fill(
            LedgerFillRecord(
                intent_id="i1",
                symbol="BTC/AUD",
                side="buy",
                quantity=1.0,
                price=50000.0,
                strategy_id="s1",
                venue="kraken",
                fees=10.0,
                slippage_bps=5.0,
                execution_alpha_bps=1.5,
            )
        )
        with open(incidents_file.name, "w", encoding="utf-8") as f:
            f.write(json.dumps({"severity": "WARNING"}) + "\n")
        ReplayAuditStore(replay_file.name).append(
            ReplayAudit(ts="2026-04-02T00:00:00+00:00", run_id="run1", status="OK", mismatch_count=0, notes="")
        )

        review = run_proving_review(
            run_id="run1",
            trade_ledger=ledger,
            incidents_path=incidents_file.name,
            replay_audit_path=replay_file.name,
            report_path=report_file.name,
        )
        assert review.trade_count == 1
        assert review.replay_ok is True
        payload = json.loads(open(report_file.name, "r", encoding="utf-8").read())
        assert payload["run_id"] == "run1"
        assert payload["promotion_decision"] in {"GO", "HOLD", "NO_GO"}
