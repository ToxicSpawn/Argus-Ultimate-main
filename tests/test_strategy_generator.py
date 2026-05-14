"""Tests for strategy generator — GP-invented trading strategies."""
import unittest
import numpy as np
from core.strategy_generator import (
    StrategyGenerator, GeneratedStrategy, TreeNode, NodeType,
    evaluate_tree, backtest_generated_strategy,
    _random_condition, _random_exit_rule, _subtree_crossover,
    _point_mutation, _hoist_mutation,
)
import random


class TestTreeNode(unittest.TestCase):
    def test_const_node(self):
        node = TreeNode(node_type=NodeType.CONST, value=42.0)
        self.assertEqual(node.depth(), 1)
        self.assertEqual(node.size(), 1)
        self.assertIn("42.00", node.to_string())

    def test_indicator_node(self):
        child = TreeNode(node_type=NodeType.CLOSE)
        node = TreeNode(node_type=NodeType.RSI, children=[child], period=14)
        self.assertEqual(node.depth(), 2)
        self.assertIn("RSI", node.to_string())

    def test_comparison_node(self):
        left = TreeNode(node_type=NodeType.RSI, children=[TreeNode(node_type=NodeType.CLOSE)], period=14)
        right = TreeNode(node_type=NodeType.CONST, value=30.0)
        node = TreeNode(node_type=NodeType.LT, children=[left, right])
        s = node.to_string()
        self.assertIn("LT", s)
        self.assertIn("RSI", s)

    def test_copy_is_deep(self):
        node = TreeNode(node_type=NodeType.AND, children=[
            TreeNode(node_type=NodeType.CONST, value=1.0),
            TreeNode(node_type=NodeType.CONST, value=2.0),
        ])
        copy = node.copy()
        copy.children[0].value = 99.0
        self.assertAlmostEqual(node.children[0].value, 1.0)


class TestEvaluateTree(unittest.TestCase):
    def setUp(self):
        self.T = 200
        t = np.arange(self.T, dtype=float)
        self.data = {
            "close": 100 + 10 * np.sin(t * 0.1) + np.random.randn(self.T) * 0.5,
            "high": 100 + 10 * np.sin(t * 0.1) + 2,
            "low": 100 + 10 * np.sin(t * 0.1) - 2,
            "volume": np.random.uniform(1e5, 1e6, self.T),
        }
        self.data["open"] = self.data["close"] * 0.999

    def test_close_returns_array(self):
        node = TreeNode(node_type=NodeType.CLOSE)
        result = evaluate_tree(node, self.data)
        self.assertEqual(len(result), self.T)

    def test_rsi_returns_bounded(self):
        node = TreeNode(node_type=NodeType.RSI, children=[TreeNode(node_type=NodeType.CLOSE)], period=14)
        result = evaluate_tree(node, self.data)
        self.assertEqual(len(result), self.T)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 100))

    def test_sma_returns_valid(self):
        node = TreeNode(node_type=NodeType.SMA, children=[TreeNode(node_type=NodeType.CLOSE)], period=20)
        result = evaluate_tree(node, self.data)
        self.assertEqual(len(result), self.T)

    def test_comparison_returns_bool_like(self):
        rsi = TreeNode(node_type=NodeType.RSI, children=[TreeNode(node_type=NodeType.CLOSE)], period=14)
        const = TreeNode(node_type=NodeType.CONST, value=30.0)
        node = TreeNode(node_type=NodeType.LT, children=[rsi, const])
        result = evaluate_tree(node, self.data)
        self.assertTrue(np.all((result == 0) | (result == 1)))

    def test_and_logic(self):
        cond1 = TreeNode(node_type=NodeType.CONST, value=1.0)
        cond2 = TreeNode(node_type=NodeType.CONST, value=0.0)
        node = TreeNode(node_type=NodeType.AND, children=[cond1, cond2])
        result = evaluate_tree(node, self.data)
        self.assertTrue(np.all(result == 0))

    def test_or_logic(self):
        cond1 = TreeNode(node_type=NodeType.CONST, value=1.0)
        cond2 = TreeNode(node_type=NodeType.CONST, value=0.0)
        node = TreeNode(node_type=NodeType.OR, children=[cond1, cond2])
        result = evaluate_tree(node, self.data)
        self.assertTrue(np.all(result == 1))

    def test_empty_data(self):
        node = TreeNode(node_type=NodeType.CLOSE)
        result = evaluate_tree(node, {"close": np.array([])})
        self.assertEqual(len(result), 0)


class TestBacktest(unittest.TestCase):
    def test_no_signals(self):
        close = np.linspace(100, 110, 100)
        entry = np.zeros(100)
        exit_sig = np.zeros(100)
        result = backtest_generated_strategy(entry, exit_sig, close)
        self.assertEqual(result["trade_count"], 0)

    def test_single_winning_trade(self):
        close = np.linspace(100, 110, 100)
        entry = np.zeros(100)
        exit_sig = np.zeros(100)
        entry[10] = 1  # buy at bar 10
        exit_sig[90] = 1  # sell at bar 90
        result = backtest_generated_strategy(entry, exit_sig, close)
        self.assertEqual(result["trade_count"], 1)
        self.assertGreater(result["total_return"], 0)

    def test_multiple_trades(self):
        close = np.linspace(100, 120, 200)
        entry = np.zeros(200)
        exit_sig = np.zeros(200)
        entry[10] = entry[60] = entry[120] = 1
        exit_sig[40] = exit_sig[90] = exit_sig[160] = 1
        result = backtest_generated_strategy(entry, exit_sig, close)
        self.assertEqual(result["trade_count"], 3)


class TestRandomGeneration(unittest.TestCase):
    def test_random_condition(self):
        rng = random.Random(42)
        cond = _random_condition(rng, max_depth=3)
        self.assertIsInstance(cond, TreeNode)
        self.assertLessEqual(cond.depth(), 7)  # generous bound

    def test_random_exit(self):
        rng = random.Random(42)
        exit_rule = _random_exit_rule(rng)
        self.assertIsInstance(exit_rule, TreeNode)
        self.assertEqual(exit_rule.node_type, NodeType.OR)


class TestGPOperators(unittest.TestCase):
    def test_subtree_crossover(self):
        rng = random.Random(42)
        p1 = _random_condition(rng, 2)
        p2 = _random_condition(rng, 2)
        child = _subtree_crossover(p1, p2, rng)
        self.assertIsInstance(child, TreeNode)

    def test_point_mutation(self):
        rng = random.Random(42)
        tree = _random_condition(rng, 2)
        mutated = _point_mutation(tree, rng)
        self.assertIsInstance(mutated, TreeNode)

    def test_hoist_mutation(self):
        rng = random.Random(42)
        tree = _random_condition(rng, 3)
        hoisted = _hoist_mutation(tree, rng)
        self.assertIsInstance(hoisted, TreeNode)
        self.assertLessEqual(hoisted.depth(), tree.depth())


class TestStrategyGenerator(unittest.TestCase):
    def setUp(self):
        self.T = 300
        t = np.arange(self.T, dtype=float)
        self.data = {
            "close": 100 + 15 * np.sin(t * 0.05) + np.random.RandomState(42).randn(self.T) * 0.5,
            "high": 100 + 15 * np.sin(t * 0.05) + 2,
            "low": 100 + 15 * np.sin(t * 0.05) - 2,
            "volume": np.random.RandomState(42).uniform(1e5, 1e6, self.T),
        }

    def test_initialize(self):
        gen = StrategyGenerator(population_size=10, seed=42)
        gen.initialize()
        self.assertEqual(len(gen._population), 10)

    def test_evolve_one_generation(self):
        gen = StrategyGenerator(population_size=10, seed=42)
        gen.initialize()
        result = gen.evolve(self.data)
        self.assertEqual(result["generation"], 1)
        self.assertEqual(result["population_size"], 10)

    def test_evolve_multiple_generations(self):
        gen = StrategyGenerator(population_size=20, seed=42)
        gen.initialize()
        for _ in range(5):
            result = gen.evolve(self.data)
        self.assertEqual(result["generation"], 5)
        self.assertGreater(result["hall_of_fame_size"], 0)

    def test_get_best(self):
        gen = StrategyGenerator(population_size=15, seed=42)
        gen.initialize()
        gen.evolve(self.data)
        best = gen.get_best()
        self.assertIsNotNone(best)
        self.assertIsInstance(best, GeneratedStrategy)

    def test_get_top(self):
        gen = StrategyGenerator(population_size=15, seed=42)
        gen.initialize()
        gen.evolve(self.data)
        top = gen.get_top(3)
        self.assertLessEqual(len(top), 3)

    def test_strategy_to_string(self):
        gen = StrategyGenerator(population_size=10, seed=42)
        gen.initialize()
        gen.evolve(self.data)
        best = gen.get_best()
        if best:
            s = best.to_string()
            self.assertIn("ENTRY:", s)
            self.assertIn("EXIT:", s)

    def test_get_stats(self):
        gen = StrategyGenerator(population_size=10, seed=42)
        gen.initialize()
        gen.evolve(self.data)
        stats = gen.get_stats()
        self.assertIn("generation", stats)
        self.assertIn("hall_of_fame_size", stats)


if __name__ == "__main__":
    unittest.main()
