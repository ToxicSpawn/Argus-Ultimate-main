# Dynamic Kelly Integration Patch

The `_calculate_quantity` method in `unified_execution_engine.py` has been
supplemented by the `risk/kelly_integration.py` module.  The existing
confidence-scaled Kelly in `_calculate_quantity` continues to work as-is;
the new `DynamicKellySizer` provides a **rolling-window empirical Kelly**
that adapts to realised trade outcomes and the current market regime.

## How to activate

1. In `unified_execution_engine.py`, add to the top-level imports:
   ```python
   from risk.kelly_integration import kelly_qty, record_trade_pnl, get_kelly_sizer
   from core.regime_bootstrap import get_regime_manager
   ```

2. In `_calculate_quantity`, replace the fixed fraction with:
   ```python
   regime_mgr = get_regime_manager()
   regime = regime_mgr.get() if regime_mgr else "ranging"

   # Dynamic Kelly from rolling trade history
   if bool(getattr(self.config, "dynamic_kelly_sizing", True)):
       equity_aud = float(
           getattr(self.config, "current_equity_aud", None)
           or getattr(self.config, "starting_capital_aud", 1000.0)
           or 1000.0
       )
       aud_to_usd = float(getattr(self.config, "aud_to_usd", 0.65) or 0.65)
       capital_usd = equity_aud * aud_to_usd
       quantity = kelly_qty(
           capital=capital_usd,
           price=signal.entry_price,
           regime=regime,
           config=self.config,
       )
   ```

3. After recording a fill in `execute_signals`, add:
   ```python
   pnl_pct = float(rd.get("pnl", 0.0) or 0.0) / max(equity_aud * aud_to_usd, 1.0)
   record_trade_pnl(pnl_pct, config=self.config)
   ```

4. In startup / `initialize()`, call:
   ```python
   from core.regime_bootstrap import bootstrap_regime_manager
   bootstrap_regime_manager(self.config)
   ```

## RegimeManager integration

To update the regime each cycle, add to your main trading loop:
```python
from core.regime_bootstrap import get_regime_manager

regime_mgr = get_regime_manager()
if regime_mgr and regime_mgr.is_stale():
    # prices_dict: {"BTC/USDT": pd.Series, "ETH/USDT": pd.Series, ...}
    regime_mgr.update_regime(prices_dict, primary="BTC/USDT")
```

This pipes CrossAssetRegimeDetector → RegimeConsensusWeighter → AdaptiveATRStops
automatically through `RegimeManager.compute_stops(ohlcv_df, entry_price)`.
