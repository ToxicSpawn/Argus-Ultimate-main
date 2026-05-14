from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class QuarantineRecord:
    ts: str
    run_id: str
    config_hash: str
    decision: str
    rollback_tag: str
    reasons: list[str]


class ConfigQuarantineStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: QuarantineRecord) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + '\n')

    def records(self) -> list[QuarantineRecord]:
        if not self.path.exists():
            return []
        rows: list[QuarantineRecord] = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(QuarantineRecord(**json.loads(line)))
        return rows

    def latest(self) -> QuarantineRecord | None:
        rows = self.records()
        return rows[-1] if rows else None

    def is_quarantined(self, config_hash: str) -> bool:
        return any(r.config_hash == config_hash for r in self.records())

    def quarantine(
        self,
        *,
        run_id: str,
        config_hash: str,
        decision: str,
        reasons: list[str],
        rollback_tag: str = 'AUTO_ROLLBACK_RECOMMENDED',
    ) -> QuarantineRecord:
        record = QuarantineRecord(
            ts=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            run_id=str(run_id),
            config_hash=str(config_hash),
            decision=str(decision),
            rollback_tag=str(rollback_tag),
            reasons=[str(r) for r in reasons],
        )
        self.append(record)
        return record
