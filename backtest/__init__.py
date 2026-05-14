"""
backtest/__init__.py — TOMBSTONE (H04 / Batch 17)

The ``backtest/`` directory has been fully retired in Batch 17.
All files that were unique to this package have been moved to
``backtesting/`` (canonical).

The three files that existed here were exact byte-for-byte duplicates
of their ``backtesting/`` counterparts (verified by SHA comparison):

  backtest/parallel_backtest.py       → backtesting/parallel_backtest.py
  backtest/report_exporter.py         → backtesting/report_exporter.py
  backtest/unified_event_backtester.py → backtesting/unified_event_backtester.py

This stub raises ImportError to catch any remaining stale references.
Do NOT import from this package — use ``backtesting`` instead::

    from backtesting.unified_event_backtester import UnifiedEventBacktester
    from backtesting.report_exporter import ReportExporter
    from backtesting.parallel_backtest import run_parallel_backtest
"""
raise ImportError(
    "'backtest' package is fully retired — use 'backtesting' instead.\n"
    "Replace: from backtest.xxx  →  from backtesting.xxx"
)
