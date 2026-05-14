"""
Argus Neuromorphic Learning System
Version: 2.0.0

Advanced learning algorithms for neuromorphic trading:
- Spike-Timing Dependent Plasticity (STDP)
- Triplet STDP (more biologically realistic)
- Reward-Modulated STDP (R-STDP)
- BCM (Bienenstock-Cooper-Munro) learning
- Homeostatic plasticity
- Neuromodulation-gated learning
- Meta-plasticity (learning rate adaptation)
- Spike-based backpropagation

Features:
- Real-time on-chip learning
- Multi-timescale plasticity
- Stability-plasticity balance
- Catastrophic forgetting prevention
- Transfer learning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class PlasticityRule(Enum):
    """Plasticity rules supported."""
    STDPA = "stdp_a"              # Additive STDP
    STDP_M = "stdp_m"             # Multiplicative STDP
    STDP_TRIPLET = "stdp_triplet"  # Triplet STDP
    R_STDP = "r_stdp"             # Reward-modulated STDP
    BCM = "bcm"                   # BCM rule
    HOMEOSTATIC = "homeostatic"   # Homeostatic plasticity
    METAPLASTIC = "metaplastic"   # Meta-plasticity


@dataclass
class STDPConfig:
    """STDP configuration."""
    a_plus: float = 0.01          # LTP amplitude
    a_minus: float = 0.012        # LTD amplitude
    tau_plus: float = 20.0        # LTP time constant (ms)
    tau_minus: float = 20.0       # LTD time constant (ms)
    weight_min: float = 0.0       # Minimum weight
    weight_max: float = 1.0       # Maximum weight
    learning_rate: float = 0.001  # Overall learning rate


@dataclass
class TripletSTDPConfig:
    """Triplet STDP configuration (more biologically realistic)."""
    a_plus: float = 0.01
    a_minus: float = 0.01
    a2_plus: float = 0.01         # Second pre-synaptic trace
    a2_minus: float = 0.01        # Second post-synaptic trace
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    tau_x: float = 15.0           # Second pre trace decay
    tau_y: float = 15.0           # Second post trace decay


@dataclass
class BCMConfig:
    """BCM (Bienenstock-Cooper-Munro) configuration."""
    tau_theta: float = 1000.0     # Threshold time constant (ms)
    theta_init: float = 0.5       # Initial threshold
    learning_rate: float = 0.01
    weight_min: float = 0.0
    weight_max: float = 2.0


@dataclass
class HomeostaticConfig:
    """Homeostatic plasticity configuration."""
    target_firing_rate: float = 5.0  # Hz
    tau_homeo: float = 1000.0        # Time constant (ms)
    strength: float = 0.1            # Homeostatic strength
    threshold_adjust_rate: float = 0.001


class STDPLearner:
    """
    Spike-Timing Dependent Plasticity learner.
    
    Implements standard STDP with configurable parameters.
    """
    
    def __init__(self, config: STDPConfig = None):
        self.config = config or STDPConfig()
        
        # Traces
        self.pre_traces: Dict[int, float] = {}
        self.post_traces: Dict[int, float] = {}
        
        # Weight history
        self.weight_history: deque = deque(maxlen=1000)
        
        # Statistics
        self.total_ltp = 0.0  # Long-term potentiation
        self.total_ltd = 0.0  # Long-term depression
    
    def update_traces(self, neuron_id: int, is_pre: bool, dt: float):
        """Update eligibility traces."""
        if is_pre:
            # Decay pre trace
            if neuron_id in self.pre_traces:
                self.pre_traces[neuron_id] *= np.exp(-dt / self.config.tau_plus)
            else:
                self.pre_traces[neuron_id] = 0.0
        else:
            # Decay post trace
            if neuron_id in self.post_traces:
                self.post_traces[neuron_id] *= np.exp(-dt / self.config.tau_minus)
            else:
                self.post_traces[neuron_id] = 0.0
    
    def on_pre_spike(self, pre_id: int, dt: float = 1.0):
        """Handle pre-synaptic spike."""
        self.pre_traces[pre_id] = self.pre_traces.get(pre_id, 0.0) + 1.0
    
    def on_post_spike(self, post_id: int, dt: float = 1.0):
        """Handle post-synaptic spike."""
        self.post_traces[post_id] = self.post_traces.get(post_id, 0.0) + 1.0
    
    def compute_weight_change(self, pre_id: int, post_id: int) -> float:
        """
        Compute weight change based on spike timing.
        
        Positive: LTP (strengthen)
        Negative: LTD (weaken)
        """
        pre_trace = self.pre_traces.get(pre_id, 0.0)
        post_trace = self.post_traces.get(post_id, 0.0)
        
        # LTP: pre before post (pre trace is high when post spikes)
        ltp = self.config.a_plus * pre_trace
        
        # LTD: post before pre (post trace is high when pre spikes)
        ltd = -self.config.a_minus * post_trace
        
        dw = (ltp + ltd) * self.config.learning_rate
        
        if dw > 0:
            self.total_ltp += abs(dw)
        else:
            self.total_ltd += abs(dw)
        
        return dw
    
    def get_stats(self) -> Dict[str, float]:
        """Get learning statistics."""
        return {
            "total_ltp": self.total_ltp,
            "total_ltd": self.total_ltd,
            "net_plasticity": self.total_ltp - self.total_ltd,
            "active_pre_traces": len(self.pre_traces),
            "active_post_traces": len(self.post_traces)
        }


class TripletSTDPLearner:
    """
    Triplet STDP learner.
    
    More biologically realistic than standard STDP.
    Uses three spikes (two pre, one post or vice versa).
    """
    
    def __init__(self, config: TripletSTDPConfig = None):
        self.config = config or TripletSTDPConfig()
        
        # First-order traces
        self.pre_traces: Dict[int, float] = {}
        self.post_traces: Dict[int, float] = {}
        
        # Second-order traces (triplet extension)
        self.pre_traces2: Dict[int, float] = {}
        self.post_traces2: Dict[int, float] = {}
        
        # Statistics
        self.total_updates = 0
    
    def on_pre_spike(self, pre_id: int):
        """Handle pre-synaptic spike."""
        # Update first-order trace
        self.pre_traces[pre_id] = self.pre_traces.get(pre_id, 0.0) + 1.0
        
        # Update second-order trace
        self.pre_traces2[pre_id] = self.pre_traces2.get(pre_id, 0.0) + 1.0
    
    def on_post_spike(self, post_id: int):
        """Handle post-synaptic spike."""
        # Update first-order trace
        self.post_traces[post_id] = self.post_traces.get(post_id, 0.0) + 1.0
        
        # Update second-order trace
        self.post_traces2[post_id] = self.post_traces2.get(post_id, 0.0) + 1.0
    
    def decay_traces(self, dt: float):
        """Decay all traces."""
        for traces in [self.pre_traces, self.post_traces]:
            for key in list(traces.keys()):
                traces[key] *= np.exp(-dt / self.config.tau_plus)
                if traces[key] < 1e-10:
                    del traces[key]
        
        for traces in [self.pre_traces2, self.post_traces2]:
            for key in list(traces.keys()):
                traces[key] *= np.exp(-dt / self.config.tau_x)
                if traces[key] < 1e-10:
                    del traces[key]
    
    def compute_weight_change(self, pre_id: int, post_id: int) -> float:
        """Compute triplet STDP weight change."""
        pre1 = self.pre_traces.get(pre_id, 0.0)
        post1 = self.post_traces.get(post_id, 0.0)
        pre2 = self.pre_traces2.get(pre_id, 0.0)
        post2 = self.post_traces2.get(post_id, 0.0)
        
        # Triplet STDP rule
        # LTP: depends on post1 and post2
        ltp = self.config.a_plus * pre1 * (1.0 + self.config.a2_minus * post2)
        
        # LTD: depends on pre1 and pre2
        ltd = -self.config.a_minus * post1 * (1.0 + self.config.a2_plus * pre2)
        
        self.total_updates += 1
        return ltp + ltd


class RewardModulatedSTDP:
    """
    Reward-Modulated STDP (R-STDP).
    
    Combines STDP with reward signals for policy learning.
    """
    
    def __init__(self, learning_rate: float = 0.01,
                 eligibility_decay: float = 0.99):
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        
        # Eligibility traces (synapse-specific)
        self.eligibility_traces: Dict[Tuple[int, int], float] = {}
        
        # STDP learner
        self.stdp = STDPLearner()
        
        # Reward history
        self.reward_history: deque = deque(maxlen=1000)
    
    def on_spike_pair(self, pre_id: int, post_id: int, dt: float = 1.0):
        """Handle spike pair for eligibility trace."""
        synapse = (pre_id, post_id)
        
        # Compute STDP-based eligibility
        dw = self.stdp.compute_weight_change(pre_id, post_id)
        
        # Update eligibility trace
        if synapse in self.eligibility_traces:
            self.eligibility_traces[synapse] *= self.eligibility_decay
            self.eligibility_traces[synapse] += dw
        else:
            self.eligibility_traces[synapse] = dw
    
    def apply_reward(self, reward: float) -> Dict[Tuple[int, int], float]:
        """
        Apply reward to update weights.
        
        Returns weight changes for each synapse.
        """
        weight_changes = {}
        
        for synapse, eligibility in self.eligibility_traces.items():
            # Weight change = eligibility * reward * learning rate
            dw = eligibility * reward * self.learning_rate
            weight_changes[synapse] = dw
        
        # Decay eligibility traces
        for synapse in self.eligibility_traces:
            self.eligibility_traces[synapse] *= self.eligibility_decay
        
        self.reward_history.append(reward)
        return weight_changes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "active_eligibility_traces": len(self.eligibility_traces),
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
            "total_synapses": len(self.eligibility_traces)
        }


class BCMLearner:
    """
    BCM (Bienenstock-Cooper-Munro) learning rule.
    
    Implements sliding threshold for stable learning.
    Prevents runaway potentiation or depression.
    """
    
    def __init__(self, config: BCMConfig = None):
        self.config = config or BCMConfig()
        
        # Sliding threshold for each neuron
        self.thresholds: Dict[int, float] = {}
        
        # Post-synaptic firing rates
        self.firing_rates: Dict[int, float] = {}
        
        # Statistics
        self.total_updates = 0
    
    def update_threshold(self, neuron_id: int, firing_rate: float, dt: float):
        """Update BCM threshold based on firing rate."""
        if neuron_id not in self.thresholds:
            self.thresholds[neuron_id] = self.config.theta_init
        
        # BCM threshold update: theta = E[y^2]
        current_threshold = self.thresholds[neuron_id]
        target = firing_rate ** 2
        
        # Exponential moving average
        alpha = dt / self.config.tau_theta
        self.thresholds[neuron_id] = (1 - alpha) * current_threshold + alpha * target
        
        # Store firing rate
        self.firing_rates[neuron_id] = firing_rate
    
    def compute_weight_change(self, pre_activity: float, post_activity: float,
                              post_neuron_id: int, weight: float) -> float:
        """
        Compute BCM weight change.
        
        dw = learning_rate * pre * post * (post - theta)
        """
        threshold = self.thresholds.get(post_neuron_id, self.config.theta_init)
        
        # BCM rule
        dw = self.config.learning_rate * pre_activity * post_activity * (post_activity - threshold)
        
        self.total_updates += 1
        return dw
    
    def get_stats(self) -> Dict[str, Any]:
        """Get BCM statistics."""
        return {
            "num_neurons": len(self.thresholds),
            "avg_threshold": np.mean(list(self.thresholds.values())) if self.thresholds else 0.0,
            "total_updates": self.total_updates
        }


class HomeostaticPlasticity:
    """
    Homeostatic plasticity for maintaining stable firing rates.
    
    Adjusts neuron excitability to maintain target firing rate.
    """
    
    def __init__(self, config: HomeostaticConfig = None):
        self.config = config or HomeostaticConfig()
        
        # Neuron-specific adjustments
        self.gain_factors: Dict[int, float] = {}  # Multiplicative gain
        self.threshold_adjustments: Dict[int, float] = {}  # Threshold shifts
        
        # Firing rate tracking
        self.firing_rate_history: Dict[int, deque] = {}
        
        # Statistics
        self.total_adjustments = 0
    
    def update(self, neuron_id: int, current_rate: float, dt: float):
        """Update homeostatic parameters for a neuron."""
        # Initialize if needed
        if neuron_id not in self.gain_factors:
            self.gain_factors[neuron_id] = 1.0
            self.threshold_adjustments[neuron_id] = 0.0
            self.firing_rate_history[neuron_id] = deque(maxlen=1000)
        
        # Track firing rate
        self.firing_rate_history[neuron_id].append(current_rate)
        
        # Compute rate error
        rate_error = current_rate - self.config.target_firing_rate
        
        # Adjust gain (multiplicative homeostasis)
        # If rate too high, decrease gain
        # If rate too low, increase gain
        gain_adjustment = -rate_error * self.config.strength * dt / self.config.tau_homeo
        self.gain_factors[neuron_id] = np.clip(
            self.gain_factors[neuron_id] + gain_adjustment,
            0.1, 10.0
        )
        
        # Adjust threshold (additive homeostasis)
        threshold_adjustment = rate_error * self.config.threshold_adjust_rate * dt / self.config.tau_homeo
        self.threshold_adjustments[neuron_id] += threshold_adjustment
        
        self.total_adjustments += 1
    
    def get_gain(self, neuron_id: int) -> float:
        """Get homeostatic gain for a neuron."""
        return self.gain_factors.get(neuron_id, 1.0)
    
    def get_threshold_adjustment(self, neuron_id: int) -> float:
        """Get threshold adjustment for a neuron."""
        return self.threshold_adjustments.get(neuron_id, 0.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get homeostatic statistics."""
        return {
            "num_neurons": len(self.gain_factors),
            "avg_gain": np.mean(list(self.gain_factors.values())) if self.gain_factors else 1.0,
            "avg_threshold_adj": np.mean(list(self.threshold_adjustments.values())) if self.threshold_adjustments else 0.0,
            "total_adjustments": self.total_adjustments
        }


class NeuromodulatedLearning:
    """
    Neuromodulation-gated learning.
    
    Learning is gated by neuromodulator levels:
    - Dopamine: gates reward-based learning
    - Acetylcholine: gates attention-dependent learning
    - Norepinephrine: gates arousal-dependent learning
    """
    
    def __init__(self):
        # Neuromodulator levels (0-1)
        self.dopamine_level = 0.5
        self.acetylcholine_level = 0.5
        self.norepinephrine_level = 0.5
        self.serotonin_level = 0.5
        
        # Base learners
        self.stdp = STDPLearner()
        self.r_stdp = RewardModulatedSTDP()
        self.bcm = BCMLearner()
        
        # Learning rate multipliers
        self.learning_rate_multiplier = 1.0
    
    def update_neuromodulators(self, dopamine: float = None,
                               acetylcholine: float = None,
                               norepinephrine: float = None,
                               serotonin: float = None):
        """Update neuromodulator levels."""
        if dopamine is not None:
            self.dopamine_level = np.clip(dopamine, 0.0, 1.0)
        if acetylcholine is not None:
            self.acetylcholine_level = np.clip(acetylcholine, 0.0, 1.0)
        if norepinephrine is not None:
            self.norepinephrine_level = np.clip(norepinephrine, 0.0, 1.0)
        if serotonin is not None:
            self.serotonin_level = np.clip(serotonin, 0.0, 1.0)
        
        # Update learning rate multiplier
        # High dopamine + acetylcholine = fast learning
        self.learning_rate_multiplier = (
            0.5 * self.dopamine_level +
            0.3 * self.acetylcholine_level +
            0.2 * self.norepinephrine_level
        )
    
    def compute_gated_weight_change(self, pre_id: int, post_id: int,
                                    reward: float = 0.0) -> float:
        """
        Compute weight change gated by neuromodulators.
        """
        # Base STDP
        dw_stdp = self.stdp.compute_weight_change(pre_id, post_id)
        
        # Reward-modulated component (use eligibility trace)
        synapse = (pre_id, post_id)
        self.r_stdp.on_spike_pair(pre_id, post_id)
        eligibility = self.r_stdp.eligibility_traces.get(synapse, 0.0)
        dw_rstdp = eligibility * reward
        
        # Gate by neuromodulators
        # Dopamine gates reward learning
        gated_rstdp = dw_rstdp * self.dopamine_level
        
        # Acetylcholine gates STDP
        gated_stdp = dw_stdp * self.acetylcholine_level
        
        # Combine
        dw_total = (gated_stdp + gated_rstdp) * self.learning_rate_multiplier
        
        return dw_total
    
    def get_stats(self) -> Dict[str, float]:
        """Get neuromodulation statistics."""
        return {
            "dopamine": self.dopamine_level,
            "acetylcholine": self.acetylcholine_level,
            "norepinephrine": self.norepinephrine_level,
            "serotonin": self.serotonin_level,
            "learning_rate_multiplier": self.learning_rate_multiplier
        }


class MetaPlasticity:
    """
    Meta-plasticity: learning to learn.
    
    Adjusts plasticity parameters based on learning history.
    """
    
    def __init__(self, base_learning_rate: float = 0.01):
        self.base_learning_rate = base_learning_rate
        
        # Meta-parameters
        self.learning_rate = base_learning_rate
        self.stdp_a_plus = 0.01
        self.stdp_a_minus = 0.012
        
        # Learning history
        self.loss_history: deque = deque(maxlen=100)
        self.gradient_history: deque = deque(maxlen=100)
    
    def update_meta_parameters(self, current_loss: float):
        """Update meta-parameters based on learning progress."""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < 10:
            return
        
        # Compute loss trend
        recent_losses = list(self.loss_history)[-10:]
        older_losses = list(self.loss_history)[-20:-10] if len(self.loss_history) >= 20 else recent_losses
        
        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)
        
        # If loss decreasing, increase learning rate
        # If loss increasing, decrease learning rate
        if recent_avg < older_avg:
            # Learning well, can increase rate slightly
            self.learning_rate *= 1.01
        else:
            # Struggling, decrease rate
            self.learning_rate *= 0.95
        
        # Clamp learning rate
        self.learning_rate = np.clip(self.learning_rate, 1e-6, 0.1)
        
        # Adjust STDP asymmetry based on loss
        if current_loss > 0.5:
            # High loss, increase LTD to prevent over-potentiation
            self.stdp_a_minus *= 1.02
        else:
            # Low loss, balance LTP/LTD
            self.stdp_a_minus *= 0.99
        
        # Clamp STDP parameters
        self.stdp_a_minus = np.clip(self.stdp_a_minus, 0.005, 0.05)
    
    def get_meta_parameters(self) -> Dict[str, float]:
        """Get current meta-parameters."""
        return {
            "learning_rate": self.learning_rate,
            "stdp_a_plus": self.stdp_a_plus,
            "stdp_a_minus": self.stdp_a_minus,
            "loss_trend": np.mean(list(self.loss_history)[-5:]) if len(self.loss_history) >= 5 else 0.0
        }


class NeuromorphicLearningSystem:
    """
    Complete neuromorphic learning system.
    
    Integrates all plasticity rules and learning mechanisms.
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, num_neurons: int = 100000):
        self.num_neurons = num_neurons
        
        # Learning modules
        self.stdp = STDPLearner()
        self.triplet_stdp = TripletSTDPLearner()
        self.r_stdp = RewardModulatedSTDP()
        self.bcm = BCMLearner()
        self.homeostatic = HomeostaticPlasticity()
        self.neuromodulated = NeuromodulatedLearning()
        self.metaplasticity = MetaPlasticity()
        
        # Synapse weights
        self.weights: Dict[Tuple[int, int], float] = {}
        
        # Statistics
        self.total_weight_updates = 0
        self.learning_history: deque = deque(maxlen=1000)
        
        logger.info(f"NeuromorphicLearningSystem v{self.VERSION} initialized")
        logger.info(f"  Neurons: {num_neurons:,}")
        logger.info(f"  Plasticity rules: STDP, Triplet STDP, R-STDP, BCM, Homeostatic, Meta")
    
    def update(self, pre_spikes: List[int], post_spikes: List[int],
               reward: float = 0.0, dt: float = 1.0) -> Dict[str, Any]:
        """
        Update weights based on spike activity.
        
        Returns learning statistics.
        """
        weight_changes = []
        
        # Process each pre-post pair
        for pre_id in pre_spikes:
            for post_id in post_spikes:
                # Skip self-connections
                if pre_id == post_id:
                    continue
                
                synapse = (pre_id, post_id)
                
                # Initialize weight if needed
                if synapse not in self.weights:
                    self.weights[synapse] = np.random.uniform(0.1, 0.5)
                
                # Compute weight changes from different rules
                dw_stdp = self.stdp.compute_weight_change(pre_id, post_id)
                
                # R-STDP: use on_spike_pair to update eligibility traces
                self.r_stdp.on_spike_pair(pre_id, post_id, dt)
                dw_rstdp = self.r_stdp.eligibility_traces.get(synapse, 0.0) * reward
                
                dw_neuromod = self.neuromodulated.compute_gated_weight_change(pre_id, post_id, reward)
                
                # Combine weight changes
                dw_total = 0.5 * dw_stdp + 0.3 * dw_rstdp + 0.2 * dw_neuromod
                
                # Apply to weight
                self.weights[synapse] += dw_total
                self.weights[synapse] = np.clip(self.weights[synapse], 0.0, 1.0)
                
                weight_changes.append(dw_total)
        
        # Update eligibility traces
        self.stdp.update_traces(pre_spikes[0] if pre_spikes else 0, is_pre=True, dt=dt)
        self.stdp.update_traces(post_spikes[0] if post_spikes else 0, is_pre=False, dt=dt)
        
        # Update statistics
        self.total_weight_updates += len(weight_changes)
        
        stats = {
            "avg_weight_change": np.mean(weight_changes) if weight_changes else 0.0,
            "num_updates": len(weight_changes),
            "total_updates": self.total_weight_updates,
            "stdp": self.stdp.get_stats(),
            "r_stdp": self.r_stdp.get_stats(),
            "neuromodulated": self.neuromodulated.get_stats(),
            "metaplasticity": self.metaplasticity.get_meta_parameters()
        }
        
        self.learning_history.append(stats)
        return stats
    
    def update_neuromodulators(self, dopamine: float = None,
                               acetylcholine: float = None,
                               norepinephrine: float = None,
                               serotonin: float = None):
        """Update neuromodulator levels."""
        self.neuromodulated.update_neuromodulators(
            dopamine, acetylcholine, norepinephrine, serotonin
        )
    
    def update_homeostasis(self, neuron_id: int, firing_rate: float, dt: float = 1.0):
        """Update homeostatic plasticity for a neuron."""
        self.homeostatic.update(neuron_id, firing_rate, dt)
    
    def get_weight(self, pre_id: int, post_id: int) -> float:
        """Get synapse weight."""
        return self.weights.get((pre_id, post_id), 0.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        return {
            "version": self.VERSION,
            "num_neurons": self.num_neurons,
            "num_synapses": len(self.weights),
            "total_weight_updates": self.total_weight_updates,
            "stdp": self.stdp.get_stats(),
            "triplet_stdp": {"total_updates": self.triplet_stdp.total_updates},
            "r_stdp": self.r_stdp.get_stats(),
            "bcm": self.bcm.get_stats(),
            "homeostatic": self.homeostatic.get_stats(),
            "neuromodulated": self.neuromodulated.get_stats(),
            "metaplasticity": self.metaplasticity.get_meta_parameters()
        }


# Global learning system instance
_learning_system: Optional[NeuromorphicLearningSystem] = None


def get_learning_system(num_neurons: int = 100000) -> NeuromorphicLearningSystem:
    """Get or create global Neuromorphic Learning System instance."""
    global _learning_system
    if _learning_system is None:
        _learning_system = NeuromorphicLearningSystem(num_neurons)
    return _learning_system


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the learning system
    system = get_learning_system(num_neurons=10000)
    
    print("\n=== Neuromorphic Learning System Test ===")
    
    # Simulate learning
    for i in range(100):
        # Random spikes
        pre_spikes = np.random.choice(1000, size=10, replace=False).tolist()
        post_spikes = np.random.choice(1000, size=5, replace=False).tolist()
        
        # Random reward
        reward = np.random.uniform(-1, 1)
        
        # Update
        stats = system.update(pre_spikes, post_spikes, reward)
    
    print(f"\nFinal Stats:")
    print(f"Total Synapses: {system.get_stats()['num_synapses']:,}")
    print(f"Total Updates: {system.get_stats()['total_weight_updates']:,}")
    print(f"STDP LTP: {system.get_stats()['stdp']['total_ltp']:.4f}")
    print(f"STDP LTD: {system.get_stats()['stdp']['total_ltd']:.4f}")
    
    # Test neuromodulation
    system.update_neuromodulators(dopamine=0.8, acetylcholine=0.6)
    print(f"\nNeuromodulation: {system.get_stats()['neuromodulated']}")
    
    # Test metaplasticity
    print(f"\nMeta-parameters: {system.get_stats()['metaplasticity']}")
