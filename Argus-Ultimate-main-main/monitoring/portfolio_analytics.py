"""Portfolio Analytics and Reporting Engine.

Features:
- Performance metrics (Sharpe, Sortino, Calmar, etc.)
- Risk analytics (VaR, CVaR, drawdown)
- Position analytics
- Trade analytics
- P&L attribution
- Strategy comparison
- Report generation (JSON, HTML, PDF)
"""

from __future__ import annotations

import logging
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"


@dataclass
class TradeRecord:
    timestamp: float
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    pnl: float = 0.0
    strategy: str = ""


@dataclass
class PositionRecord:
    symbol: str
    qty: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    duration_secs: float
    strategy: str = ""


@dataclass
class PerformanceMetrics:
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_trade_duration_secs: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


@dataclass
class RiskMetrics:
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    beta: float = 1.0
    correlation_btc: float = 0.0
    correlation_eth: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0


class PortfolioAnalyzer:
    def __init__(self, initial_capital: float = 10000.0):
        self._initial_capital = initial_capital
        self._current_capital = initial_capital
        self._trades: List[TradeRecord] = []
        self._positions: List[PositionRecord] = []
        self._equity_curve: deque = deque(maxlen=10000)
        self._daily_returns: deque = deque(maxlen=252)
        self._benchmark_returns: deque = deque(maxlen=252)

    def add_trade(self, trade: TradeRecord) -> None:
        self._trades.append(trade)
        self._current_capital += trade.pnl

    def add_position(self, position: PositionRecord) -> None:
        self._positions.append(position)

    def record_equity(self, equity: float, timestamp: float) -> None:
        self._equity_curve.append({"equity": equity, "timestamp": timestamp})
        
        if len(self._equity_curve) >= 2:
            prev = list(self._equity_curve)[-2]["equity"]
            daily_return = (equity - prev) / prev if prev > 0 else 0
            self._daily_returns.append(daily_return)

    def record_benchmark_return(self, return_pct: float) -> None:
        self._benchmark_returns.append(return_pct)

    def calculate_performance_metrics(self) -> PerformanceMetrics:
        if not self._trades:
            return PerformanceMetrics()
        
        returns = [t.pnl / self._initial_capital * 100 for t in self._trades]
        
        total_return = (self._current_capital - self._initial_capital) / self._initial_capital * 100
        
        if len(self._equity_curve) > 1:
            days = (list(self._equity_curve)[-1]["timestamp"] - list(self._equity_curve)[0]["timestamp"]) / 86400
            annual_return = total_return / max(1, days) * 365
        else:
            annual_return = 0
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if downside_returns else 1
        sortino = annual_return / downside_std if downside_std > 0 else 0
        
        equity_curve = np.array([e["equity"] for e in self._equity_curve])
        if len(equity_curve) > 1:
            cummax = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - cummax) / cummax * 100
            max_drawdown = abs(np.min(drawdowns))
            avg_drawdown = abs(np.mean(drawdowns))
        else:
            max_drawdown = 0
            avg_drawdown = 0
        
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        winning = [t for t in self._trades if t.pnl > 0]
        losing = [t for t in self._trades if t.pnl < 0]
        
        win_rate = len(winning) / len(self._trades) * 100 if self._trades else 0
        
        total_wins = sum(t.pnl for t in winning) if winning else 0
        total_losses = abs(sum(t.pnl for t in losing)) if losing else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        avg_trade_pnl = np.mean(returns)
        best_trade = max(returns) if returns else 0
        worst_trade = min(returns) if returns else 0
        
        if self._positions:
            durations = [p.duration_secs for p in self._positions]
            avg_duration = np.mean(durations)
        else:
            avg_duration = 0
        
        return PerformanceMetrics(
            total_return_pct=total_return,
            annual_return_pct=annual_return,
            volatility_pct=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_drawdown,
            avg_drawdown_pct=avg_drawdown,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            best_trade_pct=best_trade,
            worst_trade_pct=worst_trade,
            avg_trade_duration_secs=avg_duration,
            total_trades=len(self._trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
        )

    def calculate_risk_metrics(self) -> RiskMetrics:
        returns = list(self._daily_returns)
        
        if len(returns) < 10:
            return RiskMetrics()
        
        returns_array = np.array(returns)
        
        var_95 = np.percentile(returns_array, 5) * 100
        var_99 = np.percentile(returns_array, 1) * 100
        
        cvar_95 = np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)]) * 100
        cvar_99 = np.mean(returns_array[returns_array <= np.percentile(returns_array, 1)]) * 100
        
        tail_returns = returns_array[returns_array < np.percentile(returns_array, 5)]
        tail_ratio = abs(np.mean(tail_returns) / np.percentile(returns_array, 5)) if len(tail_returns) > 0 else 1
        
        skewness = np.mean(((returns_array - np.mean(returns_array)) / np.std(returns_array)) ** 3)
        kurtosis = np.mean(((returns_array - np.mean(returns_array)) / np.std(returns_array)) ** 4) - 3
        
        return RiskMetrics(
            var_95=abs(var_95),
            var_99=abs(var_99),
            cvar_95=abs(cvar_95) if not np.isnan(cvar_95) else 0,
            cvar_99=abs(cvar_99) if not np.isnan(cvar_99) else 0,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        return list(self._equity_curve)

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[TradeRecord]:
        trades = self._trades
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        if strategy:
            trades = [t for t in trades if t.strategy == strategy]
        
        return trades


class StrategyAnalyzer:
    def __init__(self):
        self._strategy_trades: Dict[str, List[TradeRecord]] = defaultdict(list)
        self._strategy_positions: Dict[str, List[PositionRecord]] = defaultdict(list)

    def add_strategy_trade(self, strategy: str, trade: TradeRecord) -> None:
        self._strategy_trades[strategy].append(trade)

    def add_strategy_position(self, strategy: str, position: PositionRecord) -> None:
        self._strategy_positions[strategy].append(position)

    def compare_strategies(self) -> Dict[str, Dict[str, Any]]:
        comparison = {}
        
        for strategy, trades in self._strategy_trades.items():
            if not trades:
                continue
            
            analyzer = PortfolioAnalyzer()
            for trade in trades:
                analyzer.add_trade(trade)
            
            metrics = analyzer.calculate_performance_metrics()
            
            comparison[strategy] = {
                "total_trades": len(trades),
                "total_return_pct": metrics.total_return_pct,
                "sharpe_ratio": metrics.sharpe_ratio,
                "win_rate_pct": metrics.win_rate_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "profit_factor": metrics.profit_factor,
                "avg_trade_pnl": metrics.avg_trade_pnl,
            }
        
        return comparison

    def rank_strategies(
        self,
        metric: str = "sharpe_ratio",
    ) -> List[Tuple[str, float]]:
        comparison = self.compare_strategies()
        
        ranked = [
            (strategy, data.get(metric, 0))
            for strategy, data in comparison.items()
        ]
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)


class ReportGenerator:
    def __init__(self, portfolio_analyzer: PortfolioAnalyzer):
        self._analyzer = portfolio_analyzer

    def generate_json_report(self) -> str:
        performance = self._analyzer.calculate_performance_metrics()
        risk = self._analyzer.calculate_risk_metrics()
        
        report = {
            "generated_at": time.time(),
            "performance": {
                "total_return_pct": round(performance.total_return_pct, 2),
                "annual_return_pct": round(performance.annual_return_pct, 2),
                "volatility_pct": round(performance.volatility_pct, 2),
                "sharpe_ratio": round(performance.sharpe_ratio, 2),
                "sortino_ratio": round(performance.sortino_ratio, 2),
                "calmar_ratio": round(performance.calmar_ratio, 2),
                "max_drawdown_pct": round(performance.max_drawdown_pct, 2),
                "win_rate_pct": round(performance.win_rate_pct, 2),
                "profit_factor": round(performance.profit_factor, 2),
                "total_trades": performance.total_trades,
            },
            "risk": {
                "var_95": round(risk.var_95, 2),
                "var_99": round(risk.var_99, 2),
                "cvar_95": round(risk.cvar_95, 2),
                "cvar_99": round(risk.cvar_99, 2),
                "tail_ratio": round(risk.tail_ratio, 2),
                "skewness": round(risk.skewness, 2),
                "kurtosis": round(risk.kurtosis, 2),
            },
            "equity_curve": self._analyzer.get_equity_curve(),
        }
        
        return json.dumps(report, indent=2)

    def generate_html_report(self) -> str:
        performance = self._analyzer.calculate_performance_metrics()
        risk = self._analyzer.calculate_risk_metrics()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Argus Ultimate - Portfolio Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f5f5f5; border-radius: 8px; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Argus Ultimate - Portfolio Report</h1>
    <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Performance Metrics</h2>
    <div>
        <div class="metric">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'positive' if performance.total_return_pct > 0 else 'negative'}">{performance.total_return_pct:.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{performance.sharpe_ratio:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{performance.win_rate_pct:.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">-{performance.max_drawdown_pct:.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{performance.total_trades}</div>
        </div>
    </div>
    
    <h2>Risk Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>VaR (95%)</td><td>{risk.var_95:.2f}%</td></tr>
        <tr><td>VaR (99%)</td><td>{risk.var_99:.2f}%</td></tr>
        <tr><td>CVaR (95%)</td><td>{risk.cvar_95:.2f}%</td></tr>
        <tr><td>Tail Ratio</td><td>{risk.tail_ratio:.2f}</td></tr>
        <tr><td>Skewness</td><td>{risk.skewness:.2f}</td></tr>
        <tr><td>Kurtosis</td><td>{risk.kurtosis:.2f}</td></tr>
    </table>
</body>
</html>
"""
        return html

    def save_report(
        self,
        filepath: str,
        format: ReportFormat = ReportFormat.JSON,
    ) -> None:
        if format == ReportFormat.JSON:
            content = self.generate_json_report()
        elif format == ReportFormat.HTML:
            content = self.generate_html_report()
        else:
            content = self.generate_json_report()
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Report saved to {filepath}")


class RealTimeMonitor:
    def __init__(self, portfolio_analyzer: PortfolioAnalyzer):
        self._analyzer = portfolio_analyzer
        self._alerts: deque = deque(maxlen=100)
        self._thresholds = {
            "max_drawdown": 10.0,
            "daily_loss": 5.0,
            "var_breach": 3.0,
        }

    def set_threshold(self, metric: str, value: float) -> None:
        self._thresholds[metric] = value

    def check_alerts(self) -> List[str]:
        alerts = []
        performance = self._analyzer.calculate_performance_metrics()
        
        if performance.max_drawdown_pct > self._thresholds["max_drawdown"]:
            alerts.append(f"Max drawdown alert: {performance.max_drawdown_pct:.2f}%")
        
        if len(self._analyzer._daily_returns) > 0:
            daily_return = list(self._analyzer._daily_returns)[-1] * 100
            if daily_return < -self._thresholds["daily_loss"]:
                alerts.append(f"Daily loss alert: {daily_return:.2f}%")
        
        risk = self._analyzer.calculate_risk_metrics()
        if risk.var_95 > self._thresholds["var_breach"]:
            alerts.append(f"VaR breach alert: {risk.var_95:.2f}%")
        
        self._alerts.extend(alerts)
        return alerts

    def get_alert_history(self) -> List[str]:
        return list(self._alerts)


class PortfolioAnalyticsEngine:
    def __init__(self, initial_capital: float = 10000.0):
        self._portfolio = PortfolioAnalyzer(initial_capital)
        self._strategy_analyzer = StrategyAnalyzer()
        self._report_generator = ReportGenerator(self._portfolio)
        self._monitor = RealTimeMonitor(self._portfolio)

    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fee: float,
        pnl: float = 0.0,
        strategy: str = "",
    ) -> None:
        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            fee=fee,
            pnl=pnl,
            strategy=strategy,
        )
        self._portfolio.add_trade(trade)
        
        if strategy:
            self._strategy_analyzer.add_strategy_trade(strategy, trade)

    def record_position(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        current_price: float,
        duration_secs: float,
        strategy: str = "",
    ) -> None:
        pnl = (current_price - entry_price) * qty
        pnl_pct = (current_price - entry_price) / entry_price * 100
        
        position = PositionRecord(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            current_price=current_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_secs=duration_secs,
            strategy=strategy,
        )
        self._portfolio.add_position(position)
        
        if strategy:
            self._strategy_analyzer.add_strategy_position(strategy, position)

    def record_equity(self, equity: float) -> None:
        self._portfolio.record_equity(equity, time.time())

    def get_performance(self) -> PerformanceMetrics:
        return self._portfolio.calculate_performance_metrics()

    def get_risk_metrics(self) -> RiskMetrics:
        return self._portfolio.calculate_risk_metrics()

    def get_strategy_comparison(self) -> Dict[str, Dict[str, Any]]:
        return self._strategy_analyzer.compare_strategies()

    def rank_strategies(self, metric: str = "sharpe_ratio") -> List[Tuple[str, float]]:
        return self._strategy_analyzer.rank_strategies(metric)

    def generate_report(
        self,
        format: ReportFormat = ReportFormat.JSON,
    ) -> str:
        if format == ReportFormat.JSON:
            return self._report_generator.generate_json_report()
        elif format == ReportFormat.HTML:
            return self._report_generator.generate_html_report()
        return self._report_generator.generate_json_report()

    def save_report(self, filepath: str, format: ReportFormat = ReportFormat.JSON) -> None:
        self._report_generator.save_report(filepath, format)

    def check_alerts(self) -> List[str]:
        return self._monitor.check_alerts()

    def set_alert_threshold(self, metric: str, value: float) -> None:
        self._monitor.set_threshold(metric, value)
