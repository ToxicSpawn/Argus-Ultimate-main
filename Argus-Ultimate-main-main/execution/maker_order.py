"""
execution/maker_order.py — DEPRECATED TOMBSTONE  (Push 30)

MakerOrderManager has been superseded by LiveExecutor + CCXTAdapter
(Push 28a/28b) which supports 50+ exchanges via a unified interface.

This module is retained as a tombstone so any code that still imports
MakerOrderManager receives a clear DeprecationWarning with migration
guidance instead of a silent ImportError.

Migration
---------
  Before (deprecated)::

      from execution.maker_order import MakerOrderManager
      mgr = MakerOrderManager(api_key=..., api_secret=...)
      mgr.place_order(symbol, side, amount)

  After (current)::

      from argus_live.execution.live_executor import LiveExecutor
      executor = LiveExecutor.from_env(exchange="kraken", dry_run=False)
      executor.place_foc_order(symbol=symbol, side=side, amount=amount)

See also
--------
  argus_live/execution/live_executor.py
  execution/ccxt_adapter.py
  docs/migration/maker_order_to_live_executor.md
"""

import warnings


class MakerOrderManager:
    """
    DEPRECATED — Use LiveExecutor from argus_live.execution.live_executor.

    Raises DeprecationWarning on instantiation.
    Raises RuntimeError on any method call.
    """

    _MIGRATION_MSG = (
        "MakerOrderManager is deprecated and has been removed (Push 30). "
        "Use LiveExecutor instead:\n"
        "  from argus_live.execution.live_executor import LiveExecutor\n"
        "  executor = LiveExecutor.from_env(exchange='kraken', dry_run=False)"
    )

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            self._MIGRATION_MSG,
            DeprecationWarning,
            stacklevel=2,
        )
        raise RuntimeError(
            "MakerOrderManager is no longer functional. " + self._MIGRATION_MSG
        )

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"MakerOrderManager.{name}() called on tombstone. " + self._MIGRATION_MSG
        )
