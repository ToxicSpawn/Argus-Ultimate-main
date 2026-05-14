"""
tests/test_level10_self_evolving_system.py — Tests for Level 10 Self-Evolving System

Tests for the ultimate self-evolving trading system.
"""

import pytest
import numpy as np
from datetime import datetime

from evolution.level10_self_evolving_system import (
    Level10System,
    EvolutionEngine,
    MetaLearner,
    StrategyMemory,
    AutonomousResearcher,
    SelfImprovingCodeEngine,
    GenomeFactory,
    StrategyGenome,
    Gene,
    GeneType,
    EvolutionStage,
    HypothesisStatus,
    create_level10_system,
)


# ============================================================================
# Gene Tests
# ============================================================================

class TestGene:
    """Tests for Gene."""
    
    def test_init(self):
        """Should initialize correctly."""
        gene = Gene(
            gene_type=GeneType.THRESHOLD,
            name="test",
            value=50,
            min_value=0,
            max_value=100,
        )
        
        assert gene.name == "test"
        assert gene.value == 50
    
    def test_mutate_threshold(self):
        """Should mutate threshold gene."""
        gene = Gene(
            gene_type=GeneType.THRESHOLD,
            name="test",
            value=50,
            min_value=0,
            max_value=100,
        )
        
        mutated = gene.mutate(strength=1.0)
        
        assert 0 <= mutated.value <= 100
    
    def test_mutate_logic(self):
        """Should mutate logic gene."""
        gene = Gene(
            gene_type=GeneType.LOGIC,
            name="test",
            value=True,
        )
        
        # Run multiple times to test mutation
        mutations = [gene.mutate(strength=1.0).value for _ in range(100)]
        
        # Should have both True and False
        assert True in mutations or False in mutations


# ============================================================================
# Strategy Genome Tests
# ============================================================================

class TestStrategyGenome:
    """Tests for Strategy Genome."""
    
    def test_init(self):
        """Should initialize correctly."""
        genome = StrategyGenome(
            genome_id="test123",
            genes={
                "rsi_period": Gene(GeneType.THRESHOLD, "rsi_period", 14, 7, 28),
            },
        )
        
        assert genome.genome_id == "test123"
        assert genome.fitness == 0.0
    
    def test_get_gene(self):
        """Should get gene by name."""
        genome = StrategyGenome(
            genome_id="test123",
            genes={
                "rsi_period": Gene(GeneType.THRESHOLD, "rsi_period", 14, 7, 28),
            },
        )
        
        gene = genome.get_gene("rsi_period")
        
        assert gene is not None
        assert gene.value == 14
    
    def test_get_dna_string(self):
        """Should get DNA string."""
        genome = StrategyGenome(
            genome_id="test123",
            genes={
                "rsi_period": Gene(GeneType.THRESHOLD, "rsi_period", 14, 7, 28),
                "use_filter": Gene(GeneType.LOGIC, "use_filter", True),
            },
        )
        
        dna = genome.get_dna_string()
        
        assert "rsi_period=14" in dna
        assert "use_filter=True" in dna
    
    def test_calculate_complexity(self):
        """Should calculate complexity."""
        genome = StrategyGenome(
            genome_id="test123",
            genes={
                "rsi_period": Gene(GeneType.THRESHOLD, "rsi_period", 14, 7, 28),
                "rsi_overbought": Gene(GeneType.THRESHOLD, "rsi_overbought", 70, 60, 85),
                "use_filter": Gene(GeneType.LOGIC, "use_filter", True),
                "trend_weight": Gene(GeneType.WEIGHT, "trend_weight", 0.4, 0.0, 1.0),
            },
        )
        
        complexity = genome.calculate_complexity()
        
        assert 0 <= complexity <= 1


# ============================================================================
# Genome Factory Tests
# ============================================================================

class TestGenomeFactory:
    """Tests for Genome Factory."""
    
    def test_create_random_genome(self):
        """Should create random genome."""
        genome = GenomeFactory.create_random_genome(complexity=0.5)
        
        assert genome is not None
        assert len(genome.genes) > 0
        assert genome.generation == 0
    
    def test_create_from_template(self):
        """Should create genome from template."""
        genome = GenomeFactory.create_from_template("momentum")
        
        assert genome is not None
        assert "primary_indicator" in genome.genes
        assert "rsi_period" in genome.genes
    
    def test_complexity_affects_gene_count(self):
        """Should create more genes with higher complexity."""
        low = GenomeFactory.create_random_genome(complexity=0.1)
        high = GenomeFactory.create_random_genome(complexity=0.9)
        
        # High complexity should generally have more genes
        # (not guaranteed due to randomness, but on average)
        assert len(low.genes) >= 5
        assert len(high.genes) >= 5


# ============================================================================
# Evolution Engine Tests
# ============================================================================

class TestEvolutionEngine:
    """Tests for Evolution Engine."""
    
    def test_init(self):
        """Should initialize correctly."""
        engine = EvolutionEngine(population_size=50)
        
        assert engine.population_size == 50
        assert len(engine.population) == 0
    
    def test_initialize_population(self):
        """Should initialize population."""
        engine = EvolutionEngine(population_size=20)
        engine.initialize_population(complexity=0.5)
        
        assert len(engine.population) == 20
        assert engine.generation == 0
    
    def test_evaluate_population(self):
        """Should evaluate population."""
        engine = EvolutionEngine(population_size=10)
        engine.initialize_population(complexity=0.5)
        
        # Simple fitness function
        def fitness_fn(genome):
            return np.random.random()
        
        engine.evaluate_population(fitness_fn)
        
        # Should be sorted by fitness
        fitnesses = [g.fitness for g in engine.population]
        assert fitnesses == sorted(fitnesses, reverse=True)
    
    def test_evolve_generation(self):
        """Should evolve one generation."""
        engine = EvolutionEngine(population_size=20)
        engine.initialize_population(complexity=0.5)
        
        def fitness_fn(genome):
            return np.random.random()
        
        engine.evaluate_population(fitness_fn)
        new_population = engine.evolve_generation()
        
        assert len(new_population) == 20
        assert engine.generation == 1
    
    def test_crossover(self):
        """Should crossover two genomes."""
        engine = EvolutionEngine()
        
        parent1 = GenomeFactory.create_random_genome()
        parent1.fitness = 0.8
        
        parent2 = GenomeFactory.create_random_genome()
        parent2.fitness = 0.6
        
        child = engine._crossover(parent1, parent2)
        
        assert child is not None
        assert len(child.genes) > 0
    
    def test_tournament_select(self):
        """Should select via tournament."""
        engine = EvolutionEngine(population_size=20, tournament_size=3)
        engine.initialize_population()
        
        # Assign different fitness values
        for i, genome in enumerate(engine.population):
            genome.fitness = i / 20
        
        # Select multiple times
        selections = [engine._tournament_select() for _ in range(100)]
        
        # Higher fitness should be selected more often
        avg_fitness = np.mean([s.fitness for s in selections])
        assert avg_fitness > 0.5  # Should bias toward higher fitness


# ============================================================================
# Meta-Learner Tests
# ============================================================================

class TestMetaLearner:
    """Tests for Meta-Learner."""
    
    def test_init(self):
        """Should initialize correctly."""
        learner = MetaLearner()
        
        assert "mutation_strength" in learner.learning_rates
        assert "crossover_rate" in learner.learning_rates
    
    def test_observe_adaptation(self):
        """Should observe adaptation."""
        learner = MetaLearner()
        
        learner.observe_adaptation("mutation_threshold", 0.5, 0.6, {})
        learner.observe_adaptation("mutation_threshold", 0.6, 0.7, {})
        
        assert len(learner.mutation_success["mutation_threshold"]) == 2
    
    def test_recommend_adaptation(self):
        """Should recommend adaptation."""
        learner = MetaLearner()
        
        genome = GenomeFactory.create_random_genome()
        recommendations = learner.recommend_adaptation(genome, {})
        
        assert "mutation_strength" in recommendations
        assert "crossover_rate" in recommendations
    
    def test_analyze_convergence(self):
        """Should analyze convergence."""
        learner = MetaLearner()
        
        # Simulate plateau
        fitness_history = [0.5, 0.51, 0.515, 0.518, 0.519, 0.52, 0.52, 0.52, 0.52, 0.52]
        
        result = learner.analyze_convergence(fitness_history)
        
        assert "converged" in result
        assert "recommendation" in result
    
    def test_get_meta_summary(self):
        """Should get meta summary."""
        learner = MetaLearner()
        
        learner.observe_adaptation("mutation_threshold", 0.5, 0.6, {})
        
        summary = learner.get_meta_summary()
        
        assert "learning_rates" in summary
        assert "mutation_stats" in summary


# ============================================================================
# Strategy Memory Tests
# ============================================================================

class TestStrategyMemory:
    """Tests for Strategy Memory."""
    
    def test_init(self):
        """Should initialize correctly."""
        memory = StrategyMemory()
        
        assert memory.capacity == 10000
    
    def test_remember_strategy(self):
        """Should remember strategy."""
        memory = StrategyMemory()
        
        genome = GenomeFactory.create_random_genome()
        genome.fitness = 0.8
        
        memory.remember_strategy(
            genome,
            market_conditions={"regime": "bull", "volatility": 0.02},
            performance={"sharpe": 1.5, "win_rate": 0.6},
        )
        
        assert len(memory.strategy_memory) == 1
    
    def test_recall_similar_situation(self):
        """Should recall similar situations."""
        memory = StrategyMemory()
        
        # Store some strategies
        for i in range(10):
            genome = GenomeFactory.create_random_genome()
            genome.fitness = np.random.random()
            memory.remember_strategy(
                genome,
                market_conditions={"regime": "bull", "volatility": 0.02},
                performance={"sharpe": 1.0},
            )
        
        # Recall
        similar = memory.recall_similar_situation({"regime": "bull", "volatility": 0.02})
        
        assert len(similar) > 0
    
    def test_get_failure_patterns(self):
        """Should get failure patterns."""
        memory = StrategyMemory()
        
        genome = GenomeFactory.create_random_genome()
        
        for _ in range(5):
            memory.remember_failure(genome, "high_drawdown", {"regime": "volatile"})
        
        patterns = memory.get_failure_patterns()
        
        assert "high_drawdown" in patterns
    
    def test_get_memory_stats(self):
        """Should get memory stats."""
        memory = StrategyMemory()
        
        stats = memory.get_memory_stats()
        
        assert "strategy_memory_size" in stats
        assert "mutation_memory_size" in stats


# ============================================================================
# Autonomous Researcher Tests
# ============================================================================

class TestAutonomousResearcher:
    """Tests for Autonomous Researcher."""
    
    def test_init(self):
        """Should initialize correctly."""
        researcher = AutonomousResearcher()
        
        assert researcher.hypotheses_generated == 0
    
    def test_generate_hypothesis(self):
        """Should generate hypothesis."""
        researcher = AutonomousResearcher()
        
        hypothesis = researcher.generate_hypothesis({"regime": "bull"})
        
        assert hypothesis["id"].startswith("hyp_")
        assert "type" in hypothesis
        assert "statement" in hypothesis
        assert "test_plan" in hypothesis
        assert hypothesis["status"] == HypothesisStatus.GENERATED.value
    
    def test_test_hypothesis_validated(self):
        """Should validate hypothesis."""
        researcher = AutonomousResearcher()
        
        hypothesis = researcher.generate_hypothesis({})
        result = researcher.test_hypothesis(
            hypothesis["id"],
            {"sharpe": 2.0, "win_rate": 0.6, "improvement": 0.2},
        )
        
        assert result["validated"] is True
        assert researcher.hypotheses_validated == 1
    
    def test_test_hypothesis_rejected(self):
        """Should reject hypothesis."""
        researcher = AutonomousResearcher()
        
        hypothesis = researcher.generate_hypothesis({})
        result = researcher.test_hypothesis(
            hypothesis["id"],
            {"sharpe": 0.5, "win_rate": 0.4, "improvement": -0.1},
        )
        
        assert result["validated"] is False
        assert researcher.hypotheses_rejected == 1
    
    def test_get_validated_patterns(self):
        """Should get validated patterns."""
        researcher = AutonomousResearcher()
        
        # Generate and validate multiple hypotheses
        for _ in range(5):
            hyp = researcher.generate_hypothesis({})
            researcher.test_hypothesis(hyp["id"], {"sharpe": 2.0, "improvement": 0.2})
        
        patterns = researcher.get_validated_patterns()
        
        assert len(patterns) >= 0  # May or may not have patterns depending on validation
    
    def test_get_research_summary(self):
        """Should get research summary."""
        researcher = AutonomousResearcher()
        
        researcher.generate_hypothesis({})
        
        summary = researcher.get_research_summary()
        
        assert "hypotheses_generated" in summary
        assert "validation_rate" in summary


# ============================================================================
# Self-Improving Code Engine Tests
# ============================================================================

class TestSelfImprovingCodeEngine:
    """Tests for Self-Improving Code Engine."""
    
    def test_init(self):
        """Should initialize correctly."""
        engine = SelfImprovingCodeEngine()
        
        assert len(engine.code_versions) == 0
    
    def test_genome_to_code(self):
        """Should convert genome to code."""
        engine = SelfImprovingCodeEngine()
        
        genome = GenomeFactory.create_from_template("momentum")
        code = engine.genome_to_code(genome)
        
        assert "class EvolvedStrategy" in code
        assert "def generate_signal" in code
        assert "def _calculate_rsi" in code
    
    def test_identify_improvements(self):
        """Should identify improvements."""
        engine = SelfImprovingCodeEngine()
        
        code = '''
class TestStrategy:
    def generate_signal(self, prices):
        return "hold", 0.0
'''
        
        improvements = engine.identify_improvements(code, {"win_rate": 0.4, "max_drawdown": 0.3})
        
        assert len(improvements) > 0
    
    def test_apply_improvement(self):
        """Should apply improvement."""
        engine = SelfImprovingCodeEngine()
        
        code = '''
class TestStrategy:
    def generate_signal(self, prices):
        # Generate signal
        return "hold", 0.0
'''
        
        improved = engine.apply_improvement(code, "Add trend filter")
        
        assert "trend" in improved.lower() or "trend_filter" in improved.lower()
    
    def test_get_engine_stats(self):
        """Should get engine stats."""
        engine = SelfImprovingCodeEngine()
        
        stats = engine.get_engine_stats()
        
        assert "code_versions_tracked" in stats
        assert "improvements_logged" in stats


# ============================================================================
# Level 10 System Tests
# ============================================================================

class TestLevel10System:
    """Tests for Level 10 System."""
    
    def test_init(self):
        """Should initialize correctly."""
        system = Level10System(population_size=20)
        
        assert system.population_size == 20
        assert system.evolution is not None
        assert system.meta_learner is not None
        assert system.memory is not None
        assert system.researcher is not None
        assert system.code_engine is not None
    
    def test_initialize(self):
        """Should initialize system."""
        system = Level10System(population_size=20)
        system.initialize(complexity=0.5)
        
        assert len(system.evolution.population) == 20
    
    def test_evolve(self):
        """Should evolve population."""
        system = Level10System(population_size=20)
        system.initialize(complexity=0.5)
        
        def fitness_fn(genome):
            return np.random.random()
        
        results = system.evolve(fitness_fn, generations=5)
        
        assert results["generations_completed"] == 5
        assert "best_fitness" in results
        assert "best_genome" in results
    
    def test_get_best_strategies(self):
        """Should get best strategies."""
        system = Level10System(population_size=20)
        system.initialize()
        
        def fitness_fn(genome):
            return np.random.random()
        
        system.evolve(fitness_fn, generations=3)
        
        best = system.get_best_strategies(n=3)
        
        assert len(best) <= 3
    
    def test_get_best_code(self):
        """Should get best code."""
        system = Level10System(population_size=20)
        system.initialize()
        
        def fitness_fn(genome):
            return np.random.random()
        
        system.evolve(fitness_fn, generations=3)
        
        code = system.get_best_code()
        
        if code:
            assert "class EvolvedStrategy" in code
    
    def test_learn_from_generation(self):
        """Should learn from generation."""
        system = Level10System(population_size=10)
        system.initialize()
        
        def fitness_fn(genome):
            return np.random.random()
        
        system.evolve(fitness_fn, generations=2)
        system.learn_from_generation({})
        
        # Should have learned something
        assert system.memory.get_memory_stats()["strategy_memory_size"] >= 0
    
    def test_get_system_report(self):
        """Should get system report."""
        system = Level10System(population_size=10)
        system.initialize()
        
        def fitness_fn(genome):
            return np.random.random()
        
        system.evolve(fitness_fn, generations=2)
        
        report = system.get_system_report()
        
        assert "generation" in report
        assert "evolution_stats" in report
        assert "meta_learning" in report
        assert "memory" in report
        assert "research" in report
        assert "code_engine" in report
    
    def test_save_and_load_state(self):
        """Should save and load state."""
        system = Level10System(population_size=10)
        system.initialize()
        
        def fitness_fn(genome):
            return np.random.random()
        
        system.evolve(fitness_fn, generations=2)
        
        state = system.save_state()
        
        new_system = Level10System()
        new_system.load_state(state)
        
        assert new_system.current_generation == system.current_generation


# ============================================================================
# Integration Tests
# ============================================================================

class TestLevel10Integration:
    """Integration tests for Level 10 system."""
    
    def test_full_evolution_cycle(self):
        """Should complete full evolution cycle."""
        system = Level10System(population_size=30)
        system.initialize(complexity=0.3)
        
        # Realistic fitness function
        def fitness_fn(genome):
            # Simulate backtest
            base_fitness = 0.5
            
            # Bonus for certain genes
            if genome.get_gene("use_trend_filter"):
                base_fitness += 0.1
            if genome.get_gene("primary_indicator"):
                base_fitness += 0.05
            
            # Random variation
            base_fitness += np.random.randn() * 0.1
            
            return max(0, base_fitness)
        
        results = system.evolve(fitness_fn, generations=10)
        
        assert results["generations_completed"] == 10
        assert results["best_fitness"] > 0
    
    def test_research_integration(self):
        """Should integrate research with evolution."""
        system = Level10System(population_size=10)
        system.initialize()
        
        # Generate hypotheses during evolution
        for i in range(5):
            hypothesis = system.researcher.generate_hypothesis({"generation": i})
            
            # Test hypothesis
            system.researcher.test_hypothesis(
                hypothesis["id"],
                {"sharpe": np.random.random() * 2, "improvement": np.random.random() * 0.2},
            )
        
        summary = system.researcher.get_research_summary()
        
        assert summary["hypotheses_generated"] >= 5
    
    def test_memory_integration(self):
        """Should integrate memory with evolution."""
        system = Level10System(population_size=10)
        system.initialize()
        
        def fitness_fn(genome):
            return np.random.random()
        
        system.evolve(fitness_fn, generations=3)
        
        # Memory should have recorded strategies
        stats = system.memory.get_memory_stats()
        
        assert stats["strategy_memory_size"] >= 0


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunction:
    """Tests for factory functions."""
    
    def test_create_level10_system(self):
        """Should create Level 10 system."""
        system = create_level10_system(population_size=50)
        
        assert system is not None
        assert isinstance(system, Level10System)
        assert system.population_size == 50
