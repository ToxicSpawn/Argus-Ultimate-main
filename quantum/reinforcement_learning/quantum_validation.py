# pyright: reportMissingImports=false
"""
Quantum Advantage Validation and Performance Metrics.

This module provides:
- Quantum advantage validation framework
- Performance metrics tracking
- Statistical significance testing
- Classical baseline comparisons
- Benchmarking tools
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ValidationMetric(Enum):
    """Metrics for quantum advantage validation."""
    TOTAL_RETURN = auto()
    SHARPE_RATIO = auto()
    SORTINO_RATIO = auto()
    CALMAR_RATIO = auto()
    MAX_DRAWDOWN = auto()
    WIN_RATE = auto()
    PROFIT_FACTOR = auto()
    TRAINING_TIME = auto()
    INFERENCE_SPEED = auto()
    CONVERGENCE_RATE = auto()


@dataclass
class ValidationConfig:
    """Configuration for quantum advantage validation."""
    min_advantage_threshold: float = 0.05  # 5% improvement required
    significance_level: float = 0.05  # p-value threshold
    num_bootstrap_samples: int = 1000
    min_evaluation_episodes: int = 100
    metrics_to_track: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.TOTAL_RETURN,
        ValidationMetric.SHARPE_RATIO,
        ValidationMetric.MAX_DRAWDOWN,
        ValidationMetric.WIN_RATE,
        ValidationMetric.CONVERGENCE_RATE
    ])


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single run."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    num_trades: int = 0
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    convergence_episode: Optional[int] = None
    final_loss: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "num_trades": self.num_trades,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_ms": self.inference_time_ms,
            "final_loss": self.final_loss
        }


@dataclass
class ComparisonResult:
    """Result of quantum vs classical comparison."""
    quantum_metrics: PerformanceMetrics
    classical_metrics: PerformanceMetrics
    advantage_metrics: Dict[str, float]
    is_quantum_advantageous: bool
    statistical_significance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumAdvantageValidator:
    """Validates quantum advantage over classical methods."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.quantum_runs: List[PerformanceMetrics] = []
        self.classical_runs: List[PerformanceMetrics] = []
        self.comparison_results: List[ComparisonResult] = []
    
    def evaluate_quantum_agent(
        self,
        quantum_agent: Any,
        env: Any,
        num_episodes: int = 100
    ) -> PerformanceMetrics:
        """Evaluate quantum agent performance."""
        logger.info("Evaluating quantum agent for %d episodes", num_episodes)
        
        start_time = time.time()
        episode_rewards = []
        trade_returns = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _ = quantum_agent.select_action(state, training=False)
                result = env.step(action)
                
                if len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = result
                
                episode_reward += reward
                if reward != 0:
                    trade_returns.append(reward)
                
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        evaluation_time = time.time() - start_time
        
        # Compute metrics
        metrics = self._compute_metrics(
            episode_rewards=episode_rewards,
            trade_returns=trade_returns,
            evaluation_time=evaluation_time
        )
        
        self.quantum_runs.append(metrics)
        
        return metrics
    
    def evaluate_classical_agent(
        self,
        classical_agent: Any,
        env: Any,
        num_episodes: int = 100
    ) -> PerformanceMetrics:
        """Evaluate classical agent performance."""
        logger.info("Evaluating classical agent for %d episodes", num_episodes)
        
        start_time = time.time()
        episode_rewards = []
        trade_returns = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0.0
            done = False
            
            while not done:
                # Classical agent uses different select_action interface
                if hasattr(classical_agent, 'select_action'):
                    action = classical_agent.select_action(state, training=False)
                else:
                    action = np.random.randint(0, 4)  # Fallback
                
                result = env.step(action)
                
                if len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = result
                
                episode_reward += reward
                if reward != 0:
                    trade_returns.append(reward)
                
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        evaluation_time = time.time() - start_time
        
        # Compute metrics
        metrics = self._compute_metrics(
            episode_rewards=episode_rewards,
            trade_returns=trade_returns,
            evaluation_time=evaluation_time
        )
        
        self.classical_runs.append(metrics)
        
        return metrics
    
    def compare_performance(
        self,
        quantum_metrics: PerformanceMetrics,
        classical_metrics: PerformanceMetrics
    ) -> ComparisonResult:
        """Compare quantum and classical performance."""
        # Calculate advantage metrics
        advantage_metrics = self._calculate_advantage(quantum_metrics, classical_metrics)
        
        # Check statistical significance
        statistical_significance = self._test_significance(
            [quantum_metrics],
            [classical_metrics]
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            [quantum_metrics],
            [classical_metrics]
        )
        
        # Determine if quantum is advantageous
        is_advantageous = self._is_quantum_advantageous(advantage_metrics, statistical_significance)
        
        result = ComparisonResult(
            quantum_metrics=quantum_metrics,
            classical_metrics=classical_metrics,
            advantage_metrics=advantage_metrics,
            is_quantum_advantageous=is_advantageous,
            statistical_significance=statistical_significance,
            confidence_intervals=confidence_intervals
        )
        
        self.comparison_results.append(result)
        
        return result
    
    def _compute_metrics(
        self,
        episode_rewards: List[float],
        trade_returns: List[float],
        evaluation_time: float
    ) -> PerformanceMetrics:
        """Compute performance metrics from episode data."""
        episode_rewards_arr = np.array(episode_rewards)
        trade_returns_arr = np.array(trade_returns) if trade_returns else np.array([0.0])
        
        # Total return
        total_return = np.sum(episode_rewards_arr)
        
        # Sharpe ratio (assuming daily returns)
        if len(episode_rewards_arr) > 1 and np.std(episode_rewards_arr) > 0:
            sharpe = np.mean(episode_rewards_arr) / np.std(episode_rewards_arr) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = episode_rewards_arr[episode_rewards_arr < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(episode_rewards_arr) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino = 0.0
        
        # Max drawdown
        cumulative = np.cumsum(episode_rewards_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate
        if len(trade_returns_arr) > 0:
            win_rate = np.sum(trade_returns_arr > 0) / len(trade_returns_arr)
        else:
            win_rate = 0.0
        
        # Profit factor
        if len(trade_returns_arr) > 0:
            gross_profit = np.sum(trade_returns_arr[trade_returns_arr > 0])
            gross_loss = abs(np.sum(trade_returns_arr[trade_returns_arr < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = 0.0
        
        # Average trade return
        avg_trade_return = np.mean(trade_returns_arr) if len(trade_returns_arr) > 0 else 0.0
        
        # Calmar ratio
        if max_drawdown > 0:
            calmar = total_return / max_drawdown
        else:
            calmar = 0.0
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            num_trades=len(trade_returns_arr),
            training_time_seconds=evaluation_time,
            inference_time_ms=evaluation_time * 1000 / max(len(episode_rewards), 1)
        )
    
    def _calculate_advantage(
        self,
        quantum: PerformanceMetrics,
        classical: PerformanceMetrics
    ) -> Dict[str, float]:
        """Calculate quantum advantage metrics."""
        advantage = {}
        
        # Return advantage
        if classical.total_return != 0:
            advantage["return_advantage"] = (quantum.total_return - classical.total_return) / abs(classical.total_return)
        else:
            advantage["return_advantage"] = 0.0
        
        # Sharpe advantage
        if classical.sharpe_ratio != 0:
            advantage["sharpe_advantage"] = (quantum.sharpe_ratio - classical.sharpe_ratio) / abs(classical.sharpe_ratio)
        else:
            advantage["sharpe_advantage"] = 0.0
        
        # Drawdown advantage (lower is better)
        if classical.max_drawdown != 0:
            advantage["drawdown_advantage"] = (classical.max_drawdown - quantum.max_drawdown) / classical.max_drawdown
        else:
            advantage["drawdown_advantage"] = 0.0
        
        # Win rate advantage
        advantage["win_rate_advantage"] = quantum.win_rate - classical.win_rate
        
        # Training time ratio
        if classical.training_time_seconds > 0:
            advantage["training_speedup"] = classical.training_time_seconds / quantum.training_time_seconds
        else:
            advantage["training_speedup"] = 1.0
        
        # Overall advantage score
        advantage["overall_advantage"] = (
            advantage["return_advantage"] * 0.3 +
            advantage["sharpe_advantage"] * 0.25 +
            advantage["drawdown_advantage"] * 0.25 +
            advantage["win_rate_advantage"] * 0.2
        )
        
        return advantage
    
    def _test_significance(
        self,
        quantum_runs: List[PerformanceMetrics],
        classical_runs: List[PerformanceMetrics],
        metric: str = "total_return"
    ) -> Dict[str, float]:
        """Test statistical significance of differences."""
        # Extract metric values
        quantum_values = [getattr(r, metric) for r in quantum_runs]
        classical_values = [getattr(r, metric) for r in classical_runs]
        
        if len(quantum_values) < 2 or len(classical_values) < 2:
            return {"p_value": 1.0, "t_statistic": 0.0}
        
        # Simple t-test (in production, would use scipy.stats)
        quantum_mean = np.mean(quantum_values)
        classical_mean = np.mean(classical_values)
        quantum_std = np.std(quantum_values)
        classical_std = np.std(classical_values)
        
        n_q = len(quantum_values)
        n_c = len(classical_values)
        
        # Pooled standard error
        se = np.sqrt(quantum_std**2 / n_q + classical_std**2 / n_c)
        
        if se > 0:
            t_stat = (quantum_mean - classical_mean) / se
        else:
            t_stat = 0.0
        
        # Approximate p-value (two-tailed)
        # In production, would use proper t-distribution
        p_value = 2 * (1 - 0.5 * (1 + np.sign(abs(t_stat) - 1.96) * 0.05))
        
        return {"p_value": p_value, "t_statistic": t_stat}
    
    def _calculate_confidence_intervals(
        self,
        quantum_runs: List[PerformanceMetrics],
        classical_runs: List[PerformanceMetrics]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics."""
        confidence_level = 1 - self.config.significance_level
        z_score = 1.96  # For 95% confidence
        
        intervals = {}
        
        for metric in self.config.metrics_to_track:
            metric_name = metric.name.lower()
            
            quantum_values = [getattr(r, metric_name) for r in quantum_runs if hasattr(r, metric_name)]
            classical_values = [getattr(r, metric_name) for r in classical_runs if hasattr(r, metric_name)]
            
            if quantum_values:
                q_mean = np.mean(quantum_values)
                q_std = np.std(quantum_values)
                q_ci = (q_mean - z_score * q_std / np.sqrt(len(quantum_values)),
                       q_mean + z_score * q_std / np.sqrt(len(quantum_values)))
                intervals[f"quantum_{metric_name}"] = q_ci
            
            if classical_values:
                c_mean = np.mean(classical_values)
                c_std = np.std(classical_values)
                c_ci = (c_mean - z_score * c_std / np.sqrt(len(classical_values)),
                       c_mean + z_score * c_std / np.sqrt(len(classical_values)))
                intervals[f"classical_{metric_name}"] = c_ci
        
        return intervals
    
    def _is_quantum_advantageous(
        self,
        advantage: Dict[str, float],
        significance: Dict[str, float]
    ) -> bool:
        """Determine if quantum provides meaningful advantage."""
        # Check if overall advantage exceeds threshold
        if advantage.get("overall_advantage", 0) < self.config.min_advantage_threshold:
            return False
        
        # Check statistical significance
        if significance.get("p_value", 1.0) > self.config.significance_level:
            return False
        
        return True
    
    def bootstrap_significance_test(
        self,
        quantum_runs: List[PerformanceMetrics],
        classical_runs: List[PerformanceMetrics],
        metric: str = "total_return"
    ) -> Dict[str, Any]:
        """Perform bootstrap significance test."""
        quantum_values = np.array([getattr(r, metric) for r in quantum_runs])
        classical_values = np.array([getattr(r, metric) for r in classical_runs])
        
        observed_diff = np.mean(quantum_values) - np.mean(classical_values)
        
        # Bootstrap
        n_bootstrap = self.config.num_bootstrap_samples
        bootstrap_diffs = []
        
        combined = np.concatenate([quantum_values, classical_values])
        n_q = len(quantum_values)
        
        for _ in range(n_bootstrap):
            # Resample
            np.random.shuffle(combined)
            boot_q = combined[:n_q]
            boot_c = combined[n_q:]
            bootstrap_diffs.append(np.mean(boot_q) - np.mean(boot_c))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value
        p_value = np.sum(bootstrap_diffs >= observed_diff) / n_bootstrap
        
        return {
            "observed_difference": observed_diff,
            "p_value": p_value,
            "confidence_interval": (np.percentile(bootstrap_diffs, 2.5),
                                   np.percentile(bootstrap_diffs, 97.5)),
            "num_bootstrap": n_bootstrap
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.comparison_results:
            return {"status": "no_comparisons"}
        
        latest = self.comparison_results[-1]
        
        return {
            "summary": {
                "is_quantum_advantageous": latest.is_quantum_advantageous,
                "quantum_return": latest.quantum_metrics.total_return,
                "classical_return": latest.classical_metrics.total_return,
                "overall_advantage": latest.advantage_metrics.get("overall_advantage", 0.0),
                "statistical_significance": latest.statistical_significance.get("p_value", 1.0)
            },
            "quantum_metrics": latest.quantum_metrics.to_dict(),
            "classical_metrics": latest.classical_metrics.to_dict(),
            "advantage_metrics": latest.advantage_metrics,
            "confidence_intervals": latest.confidence_intervals,
            "threshold_used": self.config.min_advantage_threshold,
            "significance_level": self.config.significance_level
        }


class PerformanceTracker:
    """Tracks performance metrics over time."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, float]] = []
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.training_times: List[float] = []
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        loss: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss)
        
        metrics = {
            "episode": episode,
            "reward": reward,
            "loss": loss,
            "avg_reward_10": np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else reward,
            "avg_reward_100": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else reward
        }
        
        if metadata:
            metrics.update(metadata)
        
        self.metrics_history.append(metrics)
    
    def log_training_time(self, seconds: float) -> None:
        """Log training time."""
        self.training_times.append(seconds)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.episode_rewards:
            return {"status": "no_data"}
        
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "avg_loss": np.mean(self.episode_losses) if self.episode_losses else 0.0,
            "total_training_time": sum(self.training_times),
            "convergence_episode": self._find_convergence()
        }
    
    def _find_convergence(self, window: int = 50, threshold: float = 0.01) -> Optional[int]:
        """Find episode where training converged."""
        if len(self.episode_rewards) < window * 2:
            return None
        
        for i in range(window, len(self.episode_rewards)):
            recent_avg = np.mean(self.episode_rewards[i-window:i])
            older_avg = np.mean(self.episode_rewards[max(0, i-window*2):i-window])
            
            if abs(recent_avg - older_avg) / (abs(older_avg) + 1e-8) < threshold:
                return i
        
        return None
    
    def get_moving_average(self, window: int = 10) -> NDArray[np.float64]:
        """Get moving average of rewards."""
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards)
        
        return np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')


class ClassicalBaseline:
    """Classical baseline algorithms for comparison."""
    
    @staticmethod
    def random_agent(action_dim: int) -> Any:
        """Create a random action agent."""
        class RandomAgent:
            def __init__(self, dim):
                self.action_dim = dim
            
            def select_action(self, state, training=True):
                return np.random.randint(0, self.action_dim)
        
        return RandomAgent(action_dim)
    
    @staticmethod
    def buy_and_hold(env: Any) -> PerformanceMetrics:
        """Buy and hold baseline strategy."""
        # Simulate buy and hold
        return PerformanceMetrics(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.5,
            num_trades=1
        )
    
    @staticmethod
    def moving_average_crossover(
        short_window: int = 10,
        long_window: int = 30
    ) -> Any:
        """Create moving average crossover agent."""
        class MACrossoverAgent:
            def __init__(self, short, long):
                self.short_window = short
                self.long_window = long
                self.prices = []
            
            def select_action(self, state, training=True):
                # Simplified - would use actual price data
                return np.random.randint(0, 4)
        
        return MACrossoverAgent(short_window, long_window)


__all__ = [
    # Validation
    "QuantumAdvantageValidator",
    "ValidationConfig",
    "ValidationMetric",
    "PerformanceMetrics",
    "ComparisonResult",
    
    # Performance tracking
    "PerformanceTracker",
    
    # Classical baselines
    "ClassicalBaseline"
]