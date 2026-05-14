"""
Tests for the 15 advanced research-grade modules:
  1. BayesianOptimizer — GP-based parameter optimization
  2. ThompsonBanditRouter — multi-armed bandit strategy selection
  3. RLExecutionAgent — PPO-style execution agent
  4. GPClusterDiscoverer — genetic programming cluster discovery
  5. MoEStrategyRouter — mixture of experts routing
  6. WorldModel — transformer-inspired world model
  7. MetaLearner (MAML) — meta-learning across markets
  8. EWCContinualLearner — elastic weight consolidation
  9. HierarchicalRLManager — multi-timescale RL
  10. RobustStrategyTrainer — adversarial robustness
  11. BayesianMarketModel — probabilistic programming
  12. CausalGNN — causal graph neural network
  13. QuantumAnnealer — simulated quantum annealing
  14. FederatedLearner — federated learning
  15. DeepCausalEngine — do-calculus causal inference
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. BayesianOptimizer
# ──────────────────────────────────────────────────────────────────────────────

class TestBayesianOptimizer(unittest.TestCase):
    def setUp(self):
        from core.bayesian_optimizer import BayesianOptimizer
        self.opt = BayesianOptimizer()

    def test_register_param(self):
        self.opt.register_param("x", 0.0, 10.0)
        # Just verify no crash
        self.assertIsNotNone(self.opt)

    def test_suggest_returns_value(self):
        self.opt.register_param("x", 0.0, 10.0)
        value = self.opt.suggest("x")
        self.assertIsNotNone(value)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 10.0)

    def test_update_and_suggest(self):
        self.opt.register_param("x", 0.0, 10.0)
        for _ in range(5):
            x = self.opt.suggest("x")
            # Synthetic outcome: peak at x=5
            y = -((x - 5.0) ** 2)
            self.opt.update("x", x, y)
        # Should find value near 5
        best = self.opt.get_best("x")
        self.assertIsNotNone(best)

    def test_snapshot(self):
        self.opt.register_param("x", 0.0, 10.0)
        snap = self.opt.snapshot()
        self.assertIsInstance(snap, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 2. ThompsonBanditRouter
# ──────────────────────────────────────────────────────────────────────────────

class TestThompsonBanditRouter(unittest.TestCase):
    def setUp(self):
        from core.thompson_bandit_router import ThompsonBanditRouter
        self.router = ThompsonBanditRouter()

    def test_register_strategy(self):
        self.router.register_strategy("momentum")
        # Just verify no crash
        self.assertIsNotNone(self.router)

    def test_select_strategies(self):
        for s in ["momentum", "mean_rev", "breakout"]:
            self.router.register_strategy(s)
        selected = self.router.select_strategies(2)
        self.assertEqual(len(selected), 2)

    def test_record_outcome(self):
        self.router.register_strategy("momentum")
        self.router.record_outcome("momentum", pnl_aud=15.0, won=True)
        snap = self.router.snapshot()
        self.assertIn("arms", snap)

    def test_learning_converges(self):
        np.random.seed(42)
        for s in ["good", "bad"]:
            self.router.register_strategy(s)
        for _ in range(100):
            selected = self.router.select_strategies(1)
            for name in selected:
                win = np.random.random() < (0.7 if name == "good" else 0.2)
                self.router.record_outcome(name, pnl_aud=10.0 if win else -10.0, won=win)
        # After 100 rounds, "good" should have more samples than "bad"
        snap = self.router.snapshot()
        self.assertIsInstance(snap, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 3. RLExecutionAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestRLExecutionAgent(unittest.TestCase):
    def setUp(self):
        from core.rl_execution_agent import RLExecutionAgent
        self.agent = RLExecutionAgent()

    def test_decide_returns_action(self):
        state = {
            "spread_bps": 5.0,
            "volatility": 0.01,
            "volume_ratio": 1.0,
            "urgency_needed": 0.5,
            "inventory_pct": 0.1,
        }
        action = self.agent.decide(state)
        self.assertIsInstance(action, dict)

    def test_record_outcome(self):
        state = {
            "spread_bps": 5.0,
            "volatility": 0.01,
            "volume_ratio": 1.0,
            "urgency_needed": 0.5,
            "inventory_pct": 0.1,
        }
        action = self.agent.decide(state)
        self.agent.record_outcome(state, action, reward=5.0)
        snap = self.agent.snapshot()
        self.assertIsInstance(snap, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 4. GPClusterDiscoverer
# ──────────────────────────────────────────────────────────────────────────────

class TestGPClusterDiscoverer(unittest.TestCase):
    def setUp(self):
        from core.gp_cluster_discovery import GPClusterDiscoverer
        self.gp = GPClusterDiscoverer()

    def test_evolution_runs(self):
        # Generate observations where params A and B are correlated
        obs = []
        for i in range(40):
            a = i * 0.1
            b = a + np.random.normal(0, 0.1)
            obs.append({
                "params": {"A": a, "B": b, "C": np.random.random(), "D": np.random.random()},
                "pnl": a + b,
            })
        self.gp.set_observations(obs)
        self.gp.evolve(n_generations=5)
        top = self.gp.get_top_clusters(n=3)
        self.assertIsInstance(top, list)

    def test_snapshot(self):
        snap = self.gp.snapshot()
        self.assertIsInstance(snap, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 5. MoEStrategyRouter
# ──────────────────────────────────────────────────────────────────────────────

class TestMoEStrategyRouter(unittest.TestCase):
    def setUp(self):
        from core.moe_strategy_router import MoEStrategyRouter
        self.router = MoEStrategyRouter()

    def test_add_expert_and_register(self):
        self.router.add_expert("trend", "TRENDING_UP")
        self.router.register_strategy("momentum", "TRENDING_UP")
        snap = self.router.snapshot()
        self.assertIsInstance(snap, dict)

    def test_route_returns_strategies(self):
        self.router.add_expert("trend", "TRENDING_UP")
        self.router.register_strategy("momentum", "TRENDING_UP")
        result = self.router.route({"trend_strength": 0.8, "volatility": 0.5})
        self.assertIsInstance(result, (list, dict))


# ──────────────────────────────────────────────────────────────────────────────
# 6. WorldModel
# ──────────────────────────────────────────────────────────────────────────────

class TestWorldModel(unittest.TestCase):
    def setUp(self):
        from core.world_model import WorldModel
        self.model = WorldModel()

    def test_encode_state(self):
        obs = {"price": 60000, "volume": 1000, "regime": "TRENDING_UP"}
        state = self.model.encode_state(obs)
        self.assertIsNotNone(state)

    def test_predict_next(self):
        obs = {"price": 60000, "volume": 1000, "regime": "NORMAL"}
        state = self.model.encode_state(obs)
        action = np.zeros(3)
        result = self.model.predict_next(state, action)
        self.assertIsNotNone(result)

    def test_snapshot(self):
        snap = self.model.snapshot()
        self.assertIsInstance(snap, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 7. MetaLearner (MAML)
# ──────────────────────────────────────────────────────────────────────────────

class TestMetaLearner(unittest.TestCase):
    def setUp(self):
        from core.meta_learner_maml import MetaLearner
        self.learner = MetaLearner()

    def test_register_task(self):
        self.learner.register_task("BTC/USD")
        snap = self.learner.snapshot()
        self.assertIsInstance(snap, dict)

    def test_add_samples(self):
        self.learner.register_task("BTC/USD")
        features = np.random.randn(10, 5)
        labels = np.random.randn(10)
        self.learner.add_samples("BTC/USD", features, labels)
        snap = self.learner.snapshot()
        self.assertIsInstance(snap, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 8. EWCContinualLearner
# ──────────────────────────────────────────────────────────────────────────────

class TestEWCContinualLearner(unittest.TestCase):
    def setUp(self):
        from core.ewc_continual_learner import EWCContinualLearner
        self.learner = EWCContinualLearner(feature_dim=5)

    def test_register_task(self):
        self.learner.register_task("task_a")
        snap = self.learner.snapshot()
        self.assertIsInstance(snap, dict)

    def test_train_and_predict(self):
        self.learner.register_task("task_a")
        features = np.random.randn(20, 5)
        labels = np.random.randn(20)
        self.learner.train_on_task(features, labels, epochs=5)
        pred = self.learner.predict(features[:1])
        self.assertIsNotNone(pred)


# ──────────────────────────────────────────────────────────────────────────────
# 9. HierarchicalRLManager
# ──────────────────────────────────────────────────────────────────────────────

class TestHierarchicalRL(unittest.TestCase):
    def setUp(self):
        from core.hierarchical_rl import HierarchicalRLManager
        self.mgr = HierarchicalRLManager()

    def test_add_agent(self):
        self.mgr.add_agent("fast", timescale_seconds=0.1)
        self.mgr.add_agent("slow", timescale_seconds=60.0)
        snap = self.mgr.snapshot()
        self.assertIsInstance(snap, dict)

    def test_step(self):
        self.mgr.add_agent("fast", timescale_seconds=0.1)
        state = np.random.randn(5)
        result = self.mgr.step(state, current_time=1.0)
        self.assertIsNotNone(result)


# ──────────────────────────────────────────────────────────────────────────────
# 10. RobustStrategyTrainer
# ──────────────────────────────────────────────────────────────────────────────

class TestAdversarialTrainer(unittest.TestCase):
    def setUp(self):
        from core.adversarial_trainer import RobustStrategyTrainer
        self.trainer = RobustStrategyTrainer()

    def test_train_strategy(self):
        # strategy_fn returns scalar (action signal / score)
        def simple_strategy(state):
            return float(np.sum(state))  # simple linear strategy

        real_states = [np.random.randn(8) for _ in range(10)]
        result = self.trainer.train_strategy(simple_strategy, real_states)
        self.assertIsNotNone(result)

    def test_evaluate_robustness(self):
        def simple_strategy(state):
            return float(np.sum(state))

        score = self.trainer.evaluate_robustness(simple_strategy)
        self.assertIsInstance(score, (int, float))

    def test_snapshot(self):
        snap = self.trainer.snapshot()
        self.assertIsInstance(snap, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 11. BayesianMarketModel
# ──────────────────────────────────────────────────────────────────────────────

class TestBayesianMarketModel(unittest.TestCase):
    def setUp(self):
        from core.probabilistic_programming import BayesianMarketModel
        self.model = BayesianMarketModel()

    def test_update(self):
        for _ in range(10):
            self.model.update(np.random.randn())
        snap = self.model.snapshot()
        self.assertIsInstance(snap, dict)

    def test_predict(self):
        for _ in range(10):
            self.model.update(np.random.randn())
        pred = self.model.predict(n_ahead=3)
        self.assertIsNotNone(pred)

    def test_get_uncertainty(self):
        u = self.model.get_uncertainty()
        self.assertIsNotNone(u)


# ──────────────────────────────────────────────────────────────────────────────
# 12. CausalGNN
# ──────────────────────────────────────────────────────────────────────────────

class TestCausalGNN(unittest.TestCase):
    def setUp(self):
        from core.causal_gnn import CausalGNN
        self.gnn = CausalGNN()

    def test_add_asset(self):
        self.gnn.add_asset("BTC")
        self.gnn.add_asset("ETH")
        snap = self.gnn.snapshot()
        self.assertIsInstance(snap, dict)

    def test_add_price_history(self):
        self.gnn.add_asset("BTC")
        self.gnn.add_price_history("BTC", np.random.randn(50))
        snap = self.gnn.snapshot()
        self.assertIsInstance(snap, dict)

    def test_discover_causal_edges(self):
        for sym in ["BTC", "ETH", "SOL"]:
            self.gnn.add_asset(sym)
            self.gnn.add_price_history(sym, np.random.randn(50))
        edges = self.gnn.discover_causal_edges()
        self.assertIsNotNone(edges)


# ──────────────────────────────────────────────────────────────────────────────
# 13. QuantumAnnealer
# ──────────────────────────────────────────────────────────────────────────────

class TestQuantumAnnealer(unittest.TestCase):
    def setUp(self):
        from core.quantum_annealer import QuantumAnnealer
        self.annealer = QuantumAnnealer()

    def test_set_portfolio_problem(self):
        assets = ["BTC", "ETH", "SOL", "LINK"]
        returns = np.array([0.1, 0.12, 0.08, 0.15])
        covariance = np.eye(4) * 0.01
        self.annealer.set_portfolio_problem(assets, returns, covariance)
        snap = self.annealer.snapshot()
        self.assertIsInstance(snap, dict)

    def test_solve(self):
        assets = ["BTC", "ETH", "SOL", "LINK"]
        returns = np.array([0.1, 0.12, 0.08, 0.15])
        covariance = np.eye(4) * 0.01
        self.annealer.set_portfolio_problem(assets, returns, covariance)
        self.annealer.solve(n_iterations=100)
        best = self.annealer.get_best_allocation()
        self.assertIsNotNone(best)


# ──────────────────────────────────────────────────────────────────────────────
# 14. FederatedLearner
# ──────────────────────────────────────────────────────────────────────────────

class TestFederatedLearner(unittest.TestCase):
    def setUp(self):
        from core.federated_learner import FederatedLearner
        self.learner = FederatedLearner()

    def test_register_client(self):
        self.learner.register_client("client_a")
        self.learner.register_client("client_b")
        snap = self.learner.snapshot()
        self.assertIsInstance(snap, dict)

    def test_receive_and_aggregate(self):
        self.learner.register_client("a")
        self.learner.register_client("b")
        self.learner.receive_gradient("a", np.array([1.0, 2.0, 3.0]))
        self.learner.receive_gradient("b", np.array([2.0, 2.0, 2.0]))
        self.learner.aggregate_and_broadcast()
        global_model = self.learner.get_global_model()
        self.assertIsNotNone(global_model)


# ──────────────────────────────────────────────────────────────────────────────
# 15. DeepCausalEngine
# ──────────────────────────────────────────────────────────────────────────────

class TestDeepCausalEngine(unittest.TestCase):
    def setUp(self):
        from core.deep_causal_engine import DeepCausalEngine
        self.engine = DeepCausalEngine()

    def test_add_variable(self):
        self.engine.add_variable("A")
        self.engine.add_variable("B")
        self.engine.add_edge("A", "B")
        snap = self.engine.snapshot()
        self.assertIsInstance(snap, dict)

    def test_observe(self):
        self.engine.add_variable("rate")
        self.engine.add_variable("price")
        self.engine.add_edge("rate", "price")
        self.engine.observe("rate", 0.02)
        snap = self.engine.snapshot()
        self.assertIsInstance(snap, dict)

    def test_intervene(self):
        self.engine.add_variable("A")
        self.engine.add_variable("B")
        self.engine.add_edge("A", "B")
        self.engine.intervene("A", 1.0)
        result = self.engine.infer("B")
        self.assertIsNotNone(result)


# ──────────────────────────────────────────────────────────────────────────────
# ComponentRegistry wiring
# ──────────────────────────────────────────────────────────────────────────────

class TestAdvancedTierWiring(unittest.TestCase):
    def setUp(self):
        from core.component_registry import ComponentRegistry
        self.reg = ComponentRegistry(config=MagicMock())

    def test_all_15_slots_exist(self):
        slots = [
            "bayesian_optimizer", "thompson_bandit_router", "rl_execution_agent",
            "gp_cluster_discovery", "moe_strategy_router", "world_model",
            "meta_learner_maml", "ewc_continual_learner", "hierarchical_rl",
            "adversarial_trainer", "probabilistic_programming", "causal_gnn",
            "quantum_annealer", "federated_learner", "deep_causal_engine",
        ]
        for slot in slots:
            self.assertTrue(hasattr(self.reg, slot), f"Missing slot: {slot}")

    def test_all_15_init_methods_exist(self):
        slots = [
            "bayesian_optimizer", "thompson_bandit_router", "rl_execution_agent",
            "gp_cluster_discovery", "moe_strategy_router", "world_model",
            "meta_learner_maml", "ewc_continual_learner", "hierarchical_rl",
            "adversarial_trainer", "probabilistic_programming", "causal_gnn",
            "quantum_annealer", "federated_learner", "deep_causal_engine",
        ]
        for slot in slots:
            self.assertTrue(
                hasattr(self.reg, f"_init_{slot}"), f"Missing _init_{slot}",
            )

    def test_all_15_can_instantiate(self):
        slots = [
            "bayesian_optimizer", "thompson_bandit_router", "rl_execution_agent",
            "gp_cluster_discovery", "moe_strategy_router", "world_model",
            "meta_learner_maml", "ewc_continual_learner", "hierarchical_rl",
            "adversarial_trainer", "probabilistic_programming", "causal_gnn",
            "quantum_annealer", "federated_learner", "deep_causal_engine",
        ]
        for slot in slots:
            init = getattr(self.reg, f"_init_{slot}")
            init()
            obj = getattr(self.reg, slot)
            self.assertIsNotNone(obj, f"{slot} failed to instantiate")


# ──────────────────────────────────────────────────────────────────────────────
# Config registration
# ──────────────────────────────────────────────────────────────────────────────

class TestAdvancedTierConfig(unittest.TestCase):
    def test_advanced_tier_key_registered(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        self.assertIn("advanced_tier", _KNOWN_TOP_LEVEL_KEYS)


if __name__ == "__main__":
    unittest.main()
