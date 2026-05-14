"""Health-check Argus polyglot compute surfaces.

This verifies the optional language accelerators without requiring Docker or
native runtimes. Components with missing native tools must still return Python
fallback results so the trading runtime remains safe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.polyglot_engine import PolyglotEngine
from unified_language_orchestrator import TaskRequest, TaskType, get_orchestrator


def main() -> None:
    engine = PolyglotEngine()
    loaded = engine.initialize()
    status = engine.get_status()
    benchmarks = engine.benchmark(n=250)

    orchestrator = get_orchestrator({"multi_language": {"enabled": True, "endpoints": {}}})
    request = TaskRequest(
        task_type=TaskType.ORDER_BOOK_PROCESSING,
        data={"bids": [[100.0, 2.0], [99.9, 1.5]], "asks": [[100.1, 1.7], [100.2, 1.1]]},
        timeout=1.0,
    )

    order_book_result = asyncio.run(orchestrator.execute_task(request))

    report = {
        "polyglot_components_loaded": loaded,
        "status": status,
        "benchmarks": benchmarks,
        "orchestrator_order_book": {
            "language_used": order_book_result.language_used,
            "success": order_book_result.success,
            "result": order_book_result.result,
            "execution_time_ms": order_book_result.execution_time_ms,
        },
    }

    out = Path("data/polyglot_health_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
