from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from argus_live.replay.journal_validator import validate_journal
from argus_live.replay.replay_engine import ReplayEngine


def _sha256_text(text: str) -> str:
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplayAudit:
    ts: str
    run_id: str
    status: str
    mismatch_count: int
    notes: str
    journal_checksum: str = ""
    terminal_state_hash: str = ""


class ReplayAuditStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, audit: ReplayAudit) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(audit), sort_keys=True) + "\n")

    def latest(self) -> ReplayAudit | None:
        if not self.path.exists():
            return None
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return None
        raw = json.loads(lines[-1])
        return ReplayAudit(**raw)

    def latest_for_run(self, run_id: str) -> ReplayAudit | None:
        if not self.path.exists():
            return None
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in reversed(lines):
            raw = json.loads(line)
            if str(raw.get("run_id", "")) == str(run_id):
                return ReplayAudit(**raw)
        return None


def build_replay_audit(*, journal_path: str | Path, run_id: str) -> ReplayAudit:
    journal_text = Path(journal_path).read_text(encoding="utf-8") if Path(journal_path).exists() else ""
    errors = validate_journal(journal_path)
    terminal_state_hash = ""
    if not errors and journal_text:
        engine = ReplayEngine()
        engine.load_journal(journal_path)
        terminal_state_hash = _sha256_text(json.dumps(engine.replay(), sort_keys=True, separators=(",", ":")))
    return ReplayAudit(
        ts=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        run_id=run_id,
        status="FAIL" if errors else "OK",
        mismatch_count=len(errors),
        notes="; ".join(errors[:10]),
        journal_checksum=_sha256_text(journal_text),
        terminal_state_hash=terminal_state_hash,
    )
