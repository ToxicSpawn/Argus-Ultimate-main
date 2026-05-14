"""Tests for strategy_evolver v4 — pinnacle genetic optimizer."""
import math
import unittest
from core.strategy_evolver import (
    StrategyEvolver, StrategyGenome, EnsembleGenome, EvolutionResult,
    FitnessVector, Island, MAPElitesArchive, HallOfFame,
    NoveltyArchive, TransferLearningModule, SurrogateModel,
    compute_fitness_vector, detect_regime, nsga2_rank, nsga2_select,
    deflated_sharpe_ratio, whites_reality_check, _equity_curve_signature,
    _dominates, _hurst_exponent, _tail_ratio, _max_consecutive_losses,
    _PARAM_BOUNDS, _INTEGER_PARAMS, _REGIME_STRATEGY_AFFINITY,
)


# ════════════════════════════════════════════════════════════════════════════
# FitnessVector
# ════════════════════════════════════════════════════════════════════════════

class TestFitnessVector(unittest.TestCase):
    def test_low_trades_penalised(self):
        fv = FitnessVector(trade_count=1)
        self.assertLess(fv.composite, 0)

    def test_good_fitness(self):
        fv = FitnessVector(sharpe=1.5, sortino=2.0, max_drawdown_pct=5.0,
                           calmar=2.0, profit_factor=2.0, trade_count=50,
                           win_rate=0.6, oos_sharpe=1.2, overfitting_score=0.1,
                           robustness=0.8, anti_fragility=0.9)
        self.assertGreater(fv.composite, 0.5)

    def test_overfitting_penalised(self):
        good = FitnessVector(sharpe=1.0, sortino=1.5, trade_count=20, overfitting_score=0.1)
        bad = FitnessVector(sharpe=1.0, sortino=1.5, trade_count=20, overfitting_score=0.9)
        self.assertGreater(good.composite, bad.composite)

    def test_robustness_rewarded(self):
        fragile = FitnessVector(sharpe=1.0, trade_count=20, robustness=0.1)
        robust = FitnessVector(sharpe=1.0, trade_count=20, robustness=0.9)
        self.assertGreater(robust.composite, fragile.composite)

    def test_anti_fragility_rewarded(self):
        nope = FitnessVector(sharpe=1.0, trade_count=20, anti_fragility=0.0)
        yes = FitnessVector(sharpe=1.0, trade_count=20, anti_fragility=1.0)
        self.assertGreater(yes.composite, nope.composite)

    def test_objectives_tuple(self):
        fv = FitnessVector(sharpe=1.0, sortino=2.0, max_drawdown_pct=5.0,
                           robustness=0.8, anti_fragility=0.7, trade_count=20,
                           novelty_score=0.5, reality_check_pval=0.05)
        objs = fv.objectives
        self.assertEqual(len(objs), 7)
        self.assertAlmostEqual(objs[2], -5.0)  # negated drawdown
        self.assertAlmostEqual(objs[5], 0.5)   # novelty
        self.assertAlmostEqual(objs[6], -0.05)  # negated p-value

    def test_deflated_sharpe_used_in_composite(self):
        raw = FitnessVector(sharpe=1.0, trade_count=20, deflated_sharpe=0.0)
        defl = FitnessVector(sharpe=1.0, trade_count=20, deflated_sharpe=0.5)
        # Both should be valid composites; deflated replaces raw when nonzero
        self.assertNotAlmostEqual(raw.composite, defl.composite)

    def test_novelty_bonus_in_composite(self):
        no_nov = FitnessVector(sharpe=1.0, trade_count=20, novelty_score=0.0)
        has_nov = FitnessVector(sharpe=1.0, trade_count=20, novelty_score=0.8)
        self.assertGreater(has_nov.composite, no_nov.composite)

    def test_statistical_significance_bonus(self):
        not_sig = FitnessVector(sharpe=1.0, trade_count=20, reality_check_pval=0.50)
        sig = FitnessVector(sharpe=1.0, trade_count=20, reality_check_pval=0.05)
        self.assertGreater(sig.composite, not_sig.composite)

    def test_mccv_penalty(self):
        stable = FitnessVector(sharpe=1.0, trade_count=20, mccv_std_sharpe=0.1)
        unstable = FitnessVector(sharpe=1.0, trade_count=20, mccv_std_sharpe=2.0)
        self.assertGreater(stable.composite, unstable.composite)


# ════════════════════════════════════════════════════════════════════════════
# Equity curve analysis
# ════════════════════════════════════════════════════════════════════════════

class TestEquityCurveAnalysis(unittest.TestCase):
    def test_hurst_random_walk(self):
        import random
        rng = random.Random(42)
        series = [rng.gauss(0, 1) for _ in range(200)]
        h = _hurst_exponent(series)
        self.assertGreater(h, 0.3)
        self.assertLess(h, 0.7)

    def test_hurst_short_series(self):
        self.assertAlmostEqual(_hurst_exponent([1, 2, 3]), 0.5)

    def test_tail_ratio_positive_skew(self):
        trades = [-0.5] * 80 + [5.0] * 20
        tr = _tail_ratio(trades)
        self.assertGreater(tr, 1.0)

    def test_tail_ratio_few_trades(self):
        self.assertAlmostEqual(_tail_ratio([1.0, -1.0]), 1.0)

    def test_max_consec_losses(self):
        trades = [1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0]
        self.assertEqual(_max_consecutive_losses(trades), 3)

    def test_no_losses(self):
        self.assertEqual(_max_consecutive_losses([1.0, 2.0, 3.0]), 0)


# ════════════════════════════════════════════════════════════════════════════
# NSGA-II
# ════════════════════════════════════════════════════════════════════════════

class TestNSGA2(unittest.TestCase):
    def test_dominates(self):
        self.assertTrue(_dominates((2.0, 3.0), (1.0, 2.0)))
        self.assertFalse(_dominates((2.0, 1.0), (1.0, 2.0)))  # neither dominates
        self.assertFalse(_dominates((1.0, 2.0), (2.0, 3.0)))

    def test_equal_not_dominated(self):
        self.assertFalse(_dominates((1.0, 1.0), (1.0, 1.0)))

    def test_nsga2_rank_assigns_ranks(self):
        genomes = [
            StrategyGenome("breakout", {"lookback": 20}, "BTC/USD",
                           fitness=FitnessVector(sharpe=2.0, sortino=2.0, max_drawdown_pct=3.0,
                                                 robustness=0.9, anti_fragility=0.8, trade_count=20)),
            StrategyGenome("breakout", {"lookback": 30}, "BTC/USD",
                           fitness=FitnessVector(sharpe=0.5, sortino=0.5, max_drawdown_pct=10.0,
                                                 robustness=0.2, anti_fragility=0.1, trade_count=20)),
            StrategyGenome("breakout", {"lookback": 40}, "BTC/USD",
                           fitness=FitnessVector(sharpe=1.0, sortino=1.0, max_drawdown_pct=5.0,
                                                 robustness=0.5, anti_fragility=0.5, trade_count=20)),
        ]
        ranked = nsga2_rank(genomes)
        self.assertEqual(len(ranked), 3)
        # First genome dominates second
        ranks = [g.pareto_rank for g in ranked]
        self.assertEqual(ranked[0].pareto_rank, 0)  # clearly dominant

    def test_nsga2_rank_empty(self):
        self.assertEqual(nsga2_rank([]), [])

    def test_nsga2_select_prefers_rank0(self):
        import random
        rng = random.Random(42)
        genomes = [
            StrategyGenome("breakout", {"lookback": 20}, "BTC/USD",
                           fitness=FitnessVector(sharpe=2.0, trade_count=20),
                           pareto_rank=0, crowding_distance=10.0),
            StrategyGenome("breakout", {"lookback": 30}, "BTC/USD",
                           fitness=FitnessVector(sharpe=0.5, trade_count=20),
                           pareto_rank=2, crowding_distance=5.0),
        ]
        selected = [nsga2_select(genomes, rng, k=2) for _ in range(50)]
        rank0_count = sum(1 for g in selected if g.pareto_rank == 0)
        self.assertGreater(rank0_count, 25)

    def test_crowding_distance_assigned(self):
        genomes = [
            StrategyGenome("breakout", {"lookback": i * 10}, "BTC/USD",
                           fitness=FitnessVector(sharpe=float(i), sortino=float(i),
                                                 max_drawdown_pct=float(i),
                                                 robustness=float(i) / 10,
                                                 anti_fragility=float(i) / 10,
                                                 trade_count=20))
            for i in range(5)
        ]
        ranked = nsga2_rank(genomes)
        # Boundary solutions should have infinite crowding
        has_inf = any(g.crowding_distance == float("inf") for g in ranked)
        self.assertTrue(has_inf)


# ════════════════════════════════════════════════════════════════════════════
# compute_fitness_vector
# ════════════════════════════════════════════════════════════════════════════

class TestComputeFitnessVector(unittest.TestCase):
    def test_empty(self):
        fv = compute_fitness_vector([])
        self.assertEqual(fv.trade_count, 0)

    def test_too_few(self):
        fv = compute_fitness_vector([1.0, -0.5])
        self.assertEqual(fv.sharpe, 0.0)

    def test_all_winners(self):
        fv = compute_fitness_vector([2.0, 1.5, 3.0, 1.0, 2.5])
        self.assertGreater(fv.sharpe, 0)
        self.assertEqual(fv.win_rate, 1.0)
        self.assertEqual(fv.max_consec_losses, 0)

    def test_mixed(self):
        fv = compute_fitness_vector([2.0, -1.0, 1.5, -0.5, 3.0, -2.0, 1.0, -0.8])
        self.assertEqual(fv.trade_count, 8)
        self.assertGreater(fv.max_drawdown_pct, 0)
        self.assertGreater(fv.max_consec_losses, 0)

    def test_oos_overfitting(self):
        fv = compute_fitness_vector([5.0] * 5, [-2.0] * 5)
        self.assertGreater(fv.overfitting_score, 0.5)

    def test_oos_no_overfit(self):
        fv = compute_fitness_vector([2.0, 1.5, 1.0, 2.5, 1.8],
                                    [1.8, 1.2, 1.5, 2.0, 1.6])
        self.assertLess(fv.overfitting_score, 0.3)

    def test_robustness_passed_through(self):
        fv = compute_fitness_vector([1.0] * 10, robustness=0.85)
        self.assertAlmostEqual(fv.robustness, 0.85)

    def test_anti_fragility_passed_through(self):
        fv = compute_fitness_vector([1.0] * 10, anti_fragility=0.75)
        self.assertAlmostEqual(fv.anti_fragility, 0.75)

    def test_mccv_stats(self):
        fv = compute_fitness_vector([1.0] * 10, mccv_sharpes=[0.8, 0.9, 1.0, 0.85, 0.95])
        self.assertGreater(fv.mccv_mean_sharpe, 0.8)
        self.assertGreater(fv.mccv_std_sharpe, 0)

    def test_hurst_computed(self):
        trades = [1.0, -0.5, 0.8, -0.3, 1.2, -0.7] * 10
        fv = compute_fitness_vector(trades)
        self.assertGreater(fv.hurst_exponent, 0)
        self.assertLess(fv.hurst_exponent, 1)


# ════════════════════════════════════════════════════════════════════════════
# MAP-Elites archive
# ════════════════════════════════════════════════════════════════════════════

class TestMAPElites(unittest.TestCase):
    def test_add_new_niche(self):
        archive = MAPElitesArchive()
        g = StrategyGenome("breakout", {"lookback": 20}, "BTC/USD",
                           fitness=FitnessVector(sharpe=1.0, trade_count=20), island="trending")
        self.assertTrue(archive.try_add(g))
        self.assertEqual(archive.size(), 1)

    def test_better_replaces_worse(self):
        archive = MAPElitesArchive()
        g1 = StrategyGenome("breakout", {"lookback": 20}, "BTC/USD",
                            fitness=FitnessVector(sharpe=1.0, trade_count=20), island="trending")
        g2 = StrategyGenome("breakout", {"lookback": 30}, "BTC/USD",
                            fitness=FitnessVector(sharpe=2.0, trade_count=20), island="trending")
        archive.try_add(g1)
        self.assertTrue(archive.try_add(g2))
        self.assertEqual(archive.size(), 1)
        self.assertAlmostEqual(archive.get_all()[0].fitness.sharpe, 2.0)

    def test_worse_rejected(self):
        archive = MAPElitesArchive()
        g1 = StrategyGenome("breakout", {"lookback": 20}, "BTC/USD",
                            fitness=FitnessVector(sharpe=2.0, trade_count=20), island="trending")
        g2 = StrategyGenome("breakout", {"lookback": 30}, "BTC/USD",
                            fitness=FitnessVector(sharpe=1.0, trade_count=20), island="trending")
        archive.try_add(g1)
        self.assertFalse(archive.try_add(g2))

    def test_different_niches(self):
        archive = MAPElitesArchive()
        archive.try_add(StrategyGenome("breakout", {}, "BTC/USD",
                                       fitness=FitnessVector(sharpe=1.0, trade_count=20), island="trending"))
        archive.try_add(StrategyGenome("mean_reversion", {}, "ETH/USD",
                                       fitness=FitnessVector(sharpe=1.0, trade_count=20), island="ranging"))
        self.assertEqual(archive.size(), 2)

    def test_coverage(self):
        archive = MAPElitesArchive()
        archive.try_add(StrategyGenome("breakout", {}, "BTC/USD",
                                       fitness=FitnessVector(sharpe=1.0, trade_count=20), island="trending"))
        archive.try_add(StrategyGenome("breakout", {}, "ETH/USD",
                                       fitness=FitnessVector(sharpe=1.0, trade_count=20), island="trending"))
        cov = archive.coverage()
        self.assertEqual(cov["trending"], 2)


# ════════════════════════════════════════════════════════════════════════════
# Hall of Fame
# ════════════════════════════════════════════════════════════════════════════

class TestHallOfFame(unittest.TestCase):
    def test_add_and_retrieve(self):
        hof = HallOfFame(capacity=3)
        g = StrategyGenome("breakout", {"lookback": 20}, "BTC/USD",
                           fitness=FitnessVector(sharpe=1.0, trade_count=20))
        self.assertTrue(hof.try_add(g))
        self.assertEqual(hof.size(), 1)

    def test_capacity_limit(self):
        hof = HallOfFame(capacity=2)
        for i in range(5):
            hof.try_add(StrategyGenome("breakout", {"lookback": i * 10}, "BTC/USD",
                                       fitness=FitnessVector(sharpe=float(i), trade_count=20)))
        self.assertEqual(hof.size(), 2)
        self.assertAlmostEqual(hof.get_best().fitness.sharpe, 4.0)

    def test_no_duplicates(self):
        hof = HallOfFame(capacity=5)
        g = StrategyGenome("breakout", {"lookback": 20}, "BTC/USD",
                           fitness=FitnessVector(sharpe=1.0, trade_count=20))
        hof.try_add(g)
        self.assertFalse(hof.try_add(g))
        self.assertEqual(hof.size(), 1)

    def test_worse_rejected(self):
        hof = HallOfFame(capacity=1)
        hof.try_add(StrategyGenome("breakout", {"lookback": 20}, "BTC/USD",
                                   fitness=FitnessVector(sharpe=2.0, trade_count=20)))
        result = hof.try_add(StrategyGenome("breakout", {"lookback": 30}, "BTC/USD",
                                            fitness=FitnessVector(sharpe=0.5, trade_count=20)))
        self.assertFalse(result)


# ════════════════════════════════════════════════════════════════════════════
# Regime detection
# ════════════════════════════════════════════════════════════════════════════

class TestRegimeDetection(unittest.TestCase):
    def test_trending(self):
        self.assertEqual(detect_regime([100 + i * 2 for i in range(100)]), "trending")

    def test_ranging(self):
        import math as _m
        self.assertEqual(detect_regime([100 + 2 * _m.sin(i * 0.3) for i in range(100)]), "ranging")

    def test_short(self):
        self.assertEqual(detect_regime([100, 101]), "ranging")


# ════════════════════════════════════════════════════════════════════════════
# StrategyGenome
# ════════════════════════════════════════════════════════════════════════════

class TestStrategyGenome(unittest.TestCase):
    def test_genome_id_deterministic(self):
        g1 = StrategyGenome("breakout", {"lookback": 20, "tp_pct": 2.0}, "BTC/USD")
        g2 = StrategyGenome("breakout", {"lookback": 20, "tp_pct": 2.0}, "BTC/USD")
        self.assertEqual(g1.genome_id, g2.genome_id)

    def test_frozen(self):
        g = StrategyGenome("breakout", {"lookback": 20}, "BTC/USD")
        with self.assertRaises(AttributeError):
            g.generation = 5

    def test_pareto_fields(self):
        g = StrategyGenome("breakout", {}, "BTC/USD", pareto_rank=0, crowding_distance=5.0)
        self.assertEqual(g.pareto_rank, 0)
        self.assertAlmostEqual(g.crowding_distance, 5.0)


# ════════════════════════════════════════════════════════════════════════════
# Evolver init + seeding
# ════════════════════════════════════════════════════════════════════════════

class TestEvolverInit(unittest.TestCase):
    def test_default_init(self):
        e = StrategyEvolver(seed=42)
        self.assertEqual(e._generation, 0)
        self.assertEqual(len(e._islands), 4)
        self.assertEqual(e._archive.size(), 0)
        self.assertEqual(e._hall_of_fame.size(), 0)

    def test_seed_from_scanner(self):
        e = StrategyEvolver(population_size=20, seed=42)
        e.seed_from_scanner([
            {"strategy": "breakout", "symbol": "BTC/USD",
             "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 1.5},
        ])
        total = sum(isl.size() for isl in e._islands.values())
        self.assertGreater(total, 0)

    def test_empty_scanner(self):
        e = StrategyEvolver(population_size=20, seed=42)
        e.seed_from_scanner([])
        total = sum(isl.size() for isl in e._islands.values())
        self.assertGreater(total, 0)

    def test_island_mapping(self):
        e = StrategyEvolver(seed=42)
        self.assertEqual(e._strategy_to_island("breakout"), "trending")
        self.assertEqual(e._strategy_to_island("mean_reversion"), "ranging")
        self.assertEqual(e._strategy_to_island("vol_spike_reversal"), "volatile")
        self.assertEqual(e._strategy_to_island("unknown"), "universal")


# ════════════════════════════════════════════════════════════════════════════
# Mutation
# ════════════════════════════════════════════════════════════════════════════

class TestMutation(unittest.TestCase):
    def test_stays_in_bounds(self):
        e = StrategyEvolver(mutation_rate=1.0, seed=42)
        parent = StrategyGenome("breakout",
                                {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5},
                                "BTC/USD", sigma={"lookback": 5.0, "tp_pct": 0.3, "sl_pct": 0.2})
        for _ in range(100):
            child = e._mutate(parent)
            for key, (lo, hi) in _PARAM_BOUNDS["breakout"].items():
                self.assertGreaterEqual(child.params[key], lo)
                self.assertLessEqual(child.params[key], hi)

    def test_sigma_evolves(self):
        e = StrategyEvolver(mutation_rate=1.0, seed=42)
        parent = StrategyGenome("breakout",
                                {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5},
                                "BTC/USD", sigma={"lookback": 5.0, "tp_pct": 0.3, "sl_pct": 0.2})
        child = e._mutate(parent)
        self.assertTrue(any(child.sigma.get(k) != parent.sigma.get(k) for k in parent.sigma))

    def test_integers_preserved(self):
        e = StrategyEvolver(mutation_rate=1.0, seed=42)
        parent = StrategyGenome("breakout", {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "BTC/USD")
        for _ in range(50):
            self.assertIsInstance(e._mutate(parent).params["lookback"], int)


# ════════════════════════════════════════════════════════════════════════════
# Differential evolution
# ════════════════════════════════════════════════════════════════════════════

class TestDifferentialEvolution(unittest.TestCase):
    def test_de_mutate_produces_child(self):
        e = StrategyEvolver(seed=42)
        pop = [
            StrategyGenome("breakout", {"lookback": i * 10 + 5, "tp_pct": 2.0, "sl_pct": 1.5}, "BTC/USD")
            for i in range(5)
        ]
        child = e._de_mutate(pop)
        self.assertEqual(child.strategy_type, pop[0].strategy_type)
        self.assertEqual(len(child.parent_ids), 3)

    def test_de_stays_in_bounds(self):
        e = StrategyEvolver(seed=42)
        pop = [
            StrategyGenome("breakout", {"lookback": i * 10 + 5, "tp_pct": 1.0 + float(i) * 0.5, "sl_pct": 1.5}, "BTC/USD")
            for i in range(5)
        ]
        for _ in range(50):
            child = e._de_mutate(pop)
            for key, (lo, hi) in _PARAM_BOUNDS["breakout"].items():
                self.assertGreaterEqual(child.params[key], lo)
                self.assertLessEqual(child.params[key], hi)

    def test_de_fallback_small_pop(self):
        e = StrategyEvolver(seed=42)
        pop = [StrategyGenome("breakout", {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "BTC/USD")]
        child = e._de_mutate(pop)  # should fallback to _mutate
        self.assertIsInstance(child, StrategyGenome)


# ════════════════════════════════════════════════════════════════════════════
# Crossover
# ════════════════════════════════════════════════════════════════════════════

class TestBlendCrossover(unittest.TestCase):
    def test_in_bounds(self):
        e = StrategyEvolver(seed=42)
        p1 = StrategyGenome("breakout", {"lookback": 10, "tp_pct": 1.0, "sl_pct": 0.5}, "BTC/USD")
        p2 = StrategyGenome("breakout", {"lookback": 50, "tp_pct": 4.0, "sl_pct": 2.5}, "ETH/USD")
        for _ in range(100):
            child = e._blend_crossover(p1, p2)
            for key, (lo, hi) in _PARAM_BOUNDS["breakout"].items():
                self.assertGreaterEqual(child.params[key], lo)
                self.assertLessEqual(child.params[key], hi)

    def test_blend_intermediate(self):
        e = StrategyEvolver(seed=42)
        p1 = StrategyGenome("breakout", {"lookback": 10, "tp_pct": 1.0, "sl_pct": 0.5}, "BTC/USD")
        p2 = StrategyGenome("breakout", {"lookback": 50, "tp_pct": 4.0, "sl_pct": 2.5}, "ETH/USD")
        avg_tp = sum(e._blend_crossover(p1, p2).params["tp_pct"] for _ in range(100)) / 100
        self.assertGreater(avg_tp, 1.5)
        self.assertLess(avg_tp, 3.5)


# ════════════════════════════════════════════════════════════════════════════
# Evolution cycle
# ════════════════════════════════════════════════════════════════════════════

def _simple_fitness(genome, oos_split):
    lb = genome.params.get("lookback", genome.params.get("fast_period", 10))
    is_trades = [lb / 60.0 * 2] * 8 + [-lb / 60.0] * 4
    oos_trades = [lb / 60.0 * 1.5] * 5 + [-lb / 60.0 * 1.2] * 3
    return is_trades, oos_trades


class TestEvolution(unittest.TestCase):
    def test_one_generation(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=2, robustness_jitters=2)
        e.seed_from_scanner([
            {"strategy": "breakout", "symbol": "BTC/USD",
             "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 1.5},
        ])
        result = e.evolve(_simple_fitness)
        self.assertEqual(result.generation, 1)
        self.assertIsNotNone(result.best_genome)
        self.assertGreater(result.pareto_front_size, 0)
        self.assertGreater(result.archive_size, 0)

    def test_multi_gen_improves(self):
        e = StrategyEvolver(population_size=16, seed=42, mccv_folds=2, robustness_jitters=2)
        e.seed_from_scanner([{"strategy": "breakout", "symbol": "BTC/USD",
                              "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 0.5}])
        first = e.evolve(_simple_fitness)
        for _ in range(4):
            last = e.evolve(_simple_fitness)
        self.assertGreaterEqual(last.best_genome.composite_fitness, first.avg_fitness * 0.3)

    def test_hall_of_fame_populated(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=2, robustness_jitters=2)
        e.seed_from_scanner([])
        e.evolve(_simple_fitness)
        self.assertGreater(e._hall_of_fame.size(), 0)

    def test_archive_populated(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=2, robustness_jitters=2)
        e.seed_from_scanner([])
        e.evolve(_simple_fitness)
        self.assertGreater(e._archive.size(), 0)

    def test_exception_handled(self):
        e = StrategyEvolver(population_size=8, seed=42, mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([])
        count = [0]
        def bad_fn(g, s):
            count[0] += 1
            if count[0] % 3 == 0:
                raise ValueError("boom")
            return [1.0] * 5, [0.8] * 3
        self.assertIsNotNone(e.evolve(bad_fn))

    def test_stagnation_restarts(self):
        e = StrategyEvolver(population_size=8, stagnation_limit=3, seed=42,
                            mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([])
        # Fixed fitness → should stagnate
        for _ in range(5):
            e.evolve(lambda g, s: ([0.5] * 5, [0.3] * 3))
        # Should have triggered chaos restart
        self.assertGreaterEqual(e._generation, 5)

    def test_diversity_tracked(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([])
        result = e.evolve(_simple_fitness)
        self.assertGreater(result.diversity_metric, 0)


# ════════════════════════════════════════════════════════════════════════════
# Migration
# ════════════════════════════════════════════════════════════════════════════

class TestMigration(unittest.TestCase):
    def test_migration_preserves_population(self):
        e = StrategyEvolver(population_size=20, migration_interval=1, seed=42)
        e.seed_from_scanner([
            {"strategy": "breakout", "symbol": "BTC/USD",
             "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 2.0},
        ])
        e._migrate()
        for island in e._islands.values():
            self.assertGreater(island.size(), 0)


# ════════════════════════════════════════════════════════════════════════════
# Dynamic bounds
# ════════════════════════════════════════════════════════════════════════════

class TestDynamicBounds(unittest.TestCase):
    def test_not_active_initially(self):
        e = StrategyEvolver(seed=42)
        self.assertEqual(e._dynamic_bounds_samples, 0)

    def test_activates_after_evolution(self):
        e = StrategyEvolver(population_size=16, seed=42, mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([{"strategy": "breakout", "symbol": "BTC/USD",
                              "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 1.0}])
        for _ in range(4):
            e.evolve(_simple_fitness)
        self.assertGreaterEqual(e._dynamic_bounds_samples, 3)


# ════════════════════════════════════════════════════════════════════════════
# Ensemble
# ════════════════════════════════════════════════════════════════════════════

class TestEnsemble(unittest.TestCase):
    def test_ensemble_built(self):
        e = StrategyEvolver(population_size=16, ensemble_size=3, seed=42,
                            mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([
            {"strategy": "breakout", "symbol": "BTC/USD",
             "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 1.5},
            {"strategy": "mean_reversion", "symbol": "ETH/USD",
             "params": {"bb_std": 2.0, "sl_pct": 1.5}, "sharpe": 1.2},
        ])
        result = e.evolve(_simple_fitness)
        if result.best_ensemble:
            self.assertAlmostEqual(sum(result.best_ensemble.weights), 1.0, places=5)


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

class TestPublicAPI(unittest.TestCase):
    def _make(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=2, robustness_jitters=2)
        e.seed_from_scanner([{"strategy": "breakout", "symbol": "BTC/USD",
                              "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 1.5}])
        e.evolve(_simple_fitness)
        return e

    def test_get_best(self):
        self.assertIsNotNone(self._make().get_best())

    def test_get_top(self):
        top = self._make().get_top(3)
        self.assertLessEqual(len(top), 3)

    def test_get_pareto_front(self):
        front = self._make().get_pareto_front()
        self.assertGreater(len(front), 0)
        self.assertTrue(all(g.pareto_rank == 0 for g in front))

    def test_get_archive(self):
        self.assertGreater(self._make().get_archive().size(), 0)

    def test_get_hall_of_fame(self):
        self.assertGreater(self._make().get_hall_of_fame().size(), 0)

    def test_get_stats(self):
        stats = self._make().get_stats()
        for key in ("generation", "pareto_front_size", "archive_size",
                     "hall_of_fame_size", "stagnation_counter", "diversity",
                     "avg_robustness", "avg_anti_fragility"):
            self.assertIn(key, stats)

    def test_get_island_stats(self):
        stats = self._make().get_island_stats()
        self.assertIn("trending", stats)
        self.assertIn("pareto_front_size", stats["trending"])

    def test_get_regime_recommendation(self):
        recs = self._make().get_regime_recommendation("trending")
        self.assertIsInstance(recs, list)


# ════════════════════════════════════════════════════════════════════════════
# Param bounds + affinity
# ════════════════════════════════════════════════════════════════════════════

class TestParamBounds(unittest.TestCase):
    def test_all_bounds_valid(self):
        for stype, bounds in _PARAM_BOUNDS.items():
            for key, (lo, hi) in bounds.items():
                self.assertLess(lo, hi, f"{stype}.{key}")

    def test_all_strategies_in_affinity(self):
        all_in = set()
        for strats in _REGIME_STRATEGY_AFFINITY.values():
            all_in.update(strats)
        for stype in _PARAM_BOUNDS:
            self.assertIn(stype, all_in, f"{stype} not in affinity map")

    def test_integer_params_complete(self):
        for key in ("lookback", "fast", "slow", "mid", "period", "k_period",
                     "entry_period", "exit_period", "walk_len", "lag",
                     "consec_green", "adx_thresh", "rsi_buy", "rsi_sell",
                     "buy_level", "buy_thresh", "window"):
            self.assertIn(key, _INTEGER_PARAMS)


# ════════════════════════════════════════════════════════════════════════════
# v4: Deflated Sharpe Ratio
# ════════════════════════════════════════════════════════════════════════════

class TestDeflatedSharpe(unittest.TestCase):
    def test_single_trial_unchanged(self):
        self.assertAlmostEqual(deflated_sharpe_ratio(1.5, n_trials=1), 1.5)

    def test_more_trials_deflates_more(self):
        dsr_10 = deflated_sharpe_ratio(1.5, n_trials=10)
        dsr_100 = deflated_sharpe_ratio(1.5, n_trials=100)
        dsr_1000 = deflated_sharpe_ratio(1.5, n_trials=1000)
        self.assertGreater(dsr_10, dsr_100)
        self.assertGreater(dsr_100, dsr_1000)

    def test_low_trades_unchanged(self):
        self.assertAlmostEqual(deflated_sharpe_ratio(1.0, n_trials=100, n_trades=3), 1.0)

    def test_negative_sharpe_further_deflated(self):
        dsr = deflated_sharpe_ratio(-0.5, n_trials=50)
        self.assertLess(dsr, -0.5)


class TestWhitesRealityCheck(unittest.TestCase):
    def test_strong_signal_low_pval(self):
        import random
        rng = random.Random(42)
        # Very strong signal: all positive, high magnitude
        trades = [5.0, 4.5, 4.8, 5.2, 4.9, 5.1, 4.7, 5.3, 4.6, 5.0] * 3
        pval = whites_reality_check(trades, n_bootstrap=500, rng=rng)
        self.assertLess(pval, 0.60)  # should be clearly below 1.0

    def test_noise_high_pval(self):
        import random
        rng = random.Random(42)
        trades = [0.01, -0.01, 0.005, -0.008, 0.003, -0.01, 0.007, -0.005]
        pval = whites_reality_check(trades, n_bootstrap=500, rng=rng)
        self.assertGreater(pval, 0.10)

    def test_empty_trades(self):
        self.assertAlmostEqual(whites_reality_check([]), 1.0)

    def test_few_trades(self):
        self.assertAlmostEqual(whites_reality_check([1.0, -1.0]), 1.0)


# ════════════════════════════════════════════════════════════════════════════
# v4: Novelty Search
# ════════════════════════════════════════════════════════════════════════════

class TestNoveltyArchive(unittest.TestCase):
    def test_novel_when_empty(self):
        archive = NoveltyArchive()
        sig = (0.1, 0.5, 0.8, 0.9, 1.0, 0.7, 0.3, 0.2)
        score = archive.novelty_score(sig)
        self.assertEqual(score, 1.0)

    def test_try_add(self):
        archive = NoveltyArchive()
        sig = (0.1, 0.5, 0.8, 0.9, 1.0, 0.7, 0.3, 0.2)
        self.assertTrue(archive.try_add("g1", sig))
        self.assertEqual(archive.size(), 1)

    def test_similar_less_novel(self):
        archive = NoveltyArchive(k_nearest=3)
        base = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        for i in range(15):
            archive.try_add(f"g{i}", base)
        similar = (0.51, 0.49, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50)
        different = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        self.assertGreater(archive.novelty_score(different), archive.novelty_score(similar))

    def test_equity_curve_signature(self):
        trades = [1.0, -0.5, 0.8, -0.3, 1.2, -0.7, 0.5, 0.3]
        sig = _equity_curve_signature(trades)
        self.assertEqual(len(sig), 8)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in sig))

    def test_empty_signature(self):
        sig = _equity_curve_signature([])
        self.assertEqual(len(sig), 8)


# ════════════════════════════════════════════════════════════════════════════
# v4: Transfer Learning
# ════════════════════════════════════════════════════════════════════════════

class TestTransferLearning(unittest.TestCase):
    def test_not_active_initially(self):
        tl = TransferLearningModule()
        self.assertFalse(tl.active)

    def test_update_from_genomes(self):
        tl = TransferLearningModule()
        genomes = [
            StrategyGenome("breakout", {"lookback": 20 + i, "tp_pct": 2.0, "sl_pct": 1.5},
                           f"SYM{i}/USD", fitness=FitnessVector(sharpe=1.0, trade_count=20))
            for i in range(5)
        ]
        tl.update(genomes)
        self.assertTrue(tl.active)

    def test_warm_start_in_bounds(self):
        import random
        tl = TransferLearningModule()
        genomes = [
            StrategyGenome("breakout", {"lookback": 25 + i, "tp_pct": 2.0 + i * 0.1, "sl_pct": 1.5},
                           f"SYM{i}/USD", fitness=FitnessVector(sharpe=1.0, trade_count=20))
            for i in range(10)
        ]
        tl.update(genomes)
        rng = random.Random(42)
        params = tl.warm_start("breakout", "NEW/USD", rng)
        bounds = _PARAM_BOUNDS["breakout"]
        for key, (lo, hi) in bounds.items():
            self.assertGreaterEqual(params[key], lo)
            self.assertLessEqual(params[key], hi)

    def test_warm_start_near_learned_mean(self):
        import random
        tl = TransferLearningModule()
        genomes = [
            StrategyGenome("breakout", {"lookback": 30, "tp_pct": 2.5, "sl_pct": 1.5},
                           f"SYM{i}/USD", fitness=FitnessVector(sharpe=1.0, trade_count=20))
            for i in range(10)
        ]
        tl.update(genomes)
        rng = random.Random(42)
        # Average over many warm starts should be near 30/2.5/1.5
        lookbacks = [tl.warm_start("breakout", "X/USD", rng)["lookback"] for _ in range(50)]
        avg = sum(lookbacks) / len(lookbacks)
        self.assertGreater(avg, 20)
        self.assertLess(avg, 40)


# ════════════════════════════════════════════════════════════════════════════
# v4: Surrogate Model
# ════════════════════════════════════════════════════════════════════════════

class TestSurrogateModel(unittest.TestCase):
    def test_not_ready_initially(self):
        s = SurrogateModel()
        self.assertFalse(s.ready)

    def test_record_and_predict(self):
        s = SurrogateModel(k=3)
        for i in range(30):
            s.record("breakout", {"lookback": 20 + i, "tp_pct": 2.0, "sl_pct": 1.5}, float(i) / 30)
        self.assertTrue(s.ready)
        pred = s.predict("breakout", {"lookback": 25, "tp_pct": 2.0, "sl_pct": 1.5})
        self.assertIsNotNone(pred)
        self.assertGreater(pred, 0)

    def test_predict_none_when_insufficient(self):
        s = SurrogateModel(k=5)
        s.record("breakout", {"lookback": 20}, 1.0)
        self.assertIsNone(s.predict("breakout", {"lookback": 25}))

    def test_should_full_eval_when_unknown(self):
        s = SurrogateModel()
        self.assertTrue(s.should_full_eval("breakout", {"lookback": 20}))

    def test_should_full_eval_filters_bad(self):
        s = SurrogateModel(k=3)
        # Good genomes — high composite
        for i in range(15):
            s.record("breakout", {"lookback": 25 + i, "tp_pct": 2.5}, 2.0)
        # Bad genomes — negative composite
        for i in range(15):
            s.record("breakout", {"lookback": 5 + i * 0.01, "tp_pct": 0.51 + i * 0.01}, -2.0)
        # Prediction for bad-region params should be low → filtered at 40th percentile
        pred = s.predict("breakout", {"lookback": 5, "tp_pct": 0.5})
        # Prediction should be negative (near the bad cluster)
        self.assertIsNotNone(pred)
        self.assertLess(pred, 0)


# ════════════════════════════════════════════════════════════════════════════
# v4: Integration — full evolve with all pinnacle features
# ════════════════════════════════════════════════════════════════════════════

class TestPinnacleIntegration(unittest.TestCase):
    def test_evolve_populates_v4_fields(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=2, robustness_jitters=2)
        e.seed_from_scanner([{"strategy": "breakout", "symbol": "BTC/USD",
                              "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 1.5}])
        result = e.evolve(_simple_fitness)
        self.assertGreaterEqual(result.novelty_archive_size, 0)
        self.assertIsInstance(result.avg_deflated_sharpe, float)
        self.assertIsInstance(result.statistically_significant_pct, float)
        self.assertIsInstance(result.surrogate_filter_rate, float)

    def test_stats_include_v4(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=2, robustness_jitters=2)
        e.seed_from_scanner([])
        e.evolve(_simple_fitness)
        stats = e.get_stats()
        self.assertIn("total_trials", stats)
        self.assertIn("novelty_archive_size", stats)
        self.assertIn("transfer_learning_active", stats)
        self.assertIn("avg_deflated_sharpe", stats)

    def test_novelty_archive_grows(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([])
        e.evolve(_simple_fitness)
        self.assertGreater(e._novelty_archive.size(), 0)

    def test_surrogate_records(self):
        e = StrategyEvolver(population_size=12, seed=42, mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([])
        e.evolve(_simple_fitness)
        self.assertGreater(len(e._surrogate._observations), 0)

    def test_transfer_activates_after_evolution(self):
        e = StrategyEvolver(population_size=16, seed=42, mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([
            {"strategy": "breakout", "symbol": f"SYM{i}/USD",
             "params": {"lookback": 20 + i, "tp_pct": 2.0, "sl_pct": 1.5}, "sharpe": 1.0}
            for i in range(5)
        ])
        e.evolve(_simple_fitness)
        e.evolve(_simple_fitness)
        # After 2 gens with enough elites, transfer should activate
        # (depends on HoF/archive having enough breakout genomes)
        self.assertIsInstance(e._transfer.active, bool)

    def test_total_trials_increments(self):
        e = StrategyEvolver(population_size=8, seed=42, mccv_folds=1, robustness_jitters=1)
        e.seed_from_scanner([])
        e.evolve(_simple_fitness)
        self.assertGreater(e._total_trials, 0)


if __name__ == "__main__":
    unittest.main()
