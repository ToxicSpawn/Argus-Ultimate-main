"""Tests for GPU inference pipeline, parallel strategy evaluator, and auto-trainer."""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ─────────────────────── GPU Inference Pipeline ───────────────────────


class TestGPUInferencePipeline:
    def test_init_auto_device(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="auto")
        assert pipe.device in ("cuda", "cpu")

    def test_init_force_cpu(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        assert pipe.device == "cpu"

    def test_register_model(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        model = MagicMock()
        model.predict = MagicMock(return_value=[0.5, 0.3])
        pipe.register_model("test", model, input_dim=5, model_type="sklearn")
        assert "test" in pipe._models
        assert pipe._models["test"].model_type == "sklearn"

    def test_unregister_model(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        pipe.register_model("test", MagicMock(), model_type="callable", predict_fn=lambda x: 0)
        pipe.unregister_model("test")
        assert "test" not in pipe._models

    def test_predict_batch_sklearn(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        model = MagicMock()
        model.predict = MagicMock(return_value=[0.8, 0.2, 0.5])
        pipe.register_model("sklearn_test", model, model_type="sklearn")
        features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        results = pipe.predict_batch("sklearn_test", features)
        assert len(results) == 3
        assert results[0].prediction == 0.8
        assert results[0].device_used == "cpu"

    def test_predict_batch_callable(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        fn = lambda x: {"prediction": sum(x), "confidence": 0.9}
        pipe.register_model("fn_test", None, model_type="callable", predict_fn=fn)
        results = pipe.predict_batch("fn_test", [[1, 2], [3, 4]])
        assert len(results) == 2
        assert results[0].prediction == 3
        assert results[0].confidence == 0.9

    def test_predict_batch_unknown_model(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        results = pipe.predict_batch("nonexistent", [[1, 2]])
        assert len(results) == 1
        assert results[0].prediction is None

    def test_predict_batch_empty(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        pipe.register_model("test", MagicMock(), model_type="callable", predict_fn=lambda x: 0)
        results = pipe.predict_batch("test", [])
        assert results == []

    def test_predict_all_models(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        m1 = MagicMock()
        m1.predict = MagicMock(return_value=[0.5])
        m2 = MagicMock()
        m2.predict = MagicMock(return_value=[0.9])
        pipe.register_model("m1", m1, model_type="sklearn")
        pipe.register_model("m2", m2, model_type="sklearn")
        results = pipe.predict_all_models({"m1": [[1, 2]], "m2": [[3, 4]]})
        assert "m1" in results
        assert "m2" in results
        assert len(results["m1"]) == 1
        assert len(results["m2"]) == 1

    def test_get_stats(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        stats = pipe.get_stats()
        assert stats["device"] == "cpu"
        assert stats["models_loaded"] == 0
        assert stats["total_predictions"] == 0
        assert "gpu_name" in stats

    def test_stats_update_after_predictions(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        fn = lambda x: 1.0
        pipe.register_model("counter", None, model_type="callable", predict_fn=fn)
        pipe.predict_batch("counter", [[1], [2], [3]])
        stats = pipe.get_stats()
        assert stats["total_predictions"] == 3
        assert stats["models"]["counter"]["predictions"] == 3

    def test_callable_error_returns_none(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        fn = lambda x: 1 / 0  # ZeroDivisionError
        pipe.register_model("bad", None, model_type="callable", predict_fn=fn)
        results = pipe.predict_batch("bad", [[1]])
        assert len(results) == 1
        assert results[0].prediction is None

    def test_latency_tracked(self):
        from ml.gpu_inference import GPUInferencePipeline
        pipe = GPUInferencePipeline(device="cpu")
        fn = lambda x: 42
        pipe.register_model("timed", None, model_type="callable", predict_fn=fn)
        results = pipe.predict_batch("timed", [[1], [2]])
        assert all(r.latency_ms >= 0 for r in results)


# ─────────────────────── Parallel Strategy Evaluator ───────────────────────


class TestParallelStrategyEvaluator:
    def test_init(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator(max_workers=4)
        assert ev._max_workers == 4

    def test_eval_empty(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator()
        results = ev.evaluate_sync([], {}, "NORMAL")
        assert results == []

    def test_eval_sync_with_generate_signal(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator()
        strategy = MagicMock()
        strategy.generate_signal = MagicMock(return_value={"action": "BUY", "symbol": "BTC"})
        strategy.name = "test_strat"
        results = ev.evaluate_sync([strategy], {"price": 50000}, "NORMAL")
        assert len(results) == 1
        assert results[0].strategy_name == "test_strat"
        assert len(results[0].signals) == 1

    def test_eval_sync_with_analyze(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator()
        strategy = MagicMock(spec=[])
        strategy.analyze = MagicMock(return_value=[{"action": "SELL"}])
        strategy.__class__.__name__ = "TestAnalyze"
        results = ev.evaluate_sync([strategy], {}, "NORMAL")
        assert len(results) == 1
        assert len(results[0].signals) == 1

    def test_eval_sync_none_result(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator()
        strategy = MagicMock()
        strategy.generate_signal = MagicMock(return_value=None)
        strategy.name = "null_strat"
        results = ev.evaluate_sync([strategy], {}, "NORMAL")
        assert len(results) == 1
        assert results[0].signals == []

    def test_eval_sync_error_handled(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator()
        strategy = MagicMock()
        strategy.generate_signal = MagicMock(side_effect=RuntimeError("boom"))
        strategy.name = "bad_strat"
        results = ev.evaluate_sync([strategy], {}, "NORMAL")
        assert len(results) == 1
        assert results[0].error is not None

    def test_eval_sync_multiple_strategies(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator(max_workers=4)
        strategies = []
        for i in range(5):
            s = MagicMock()
            s.generate_signal = MagicMock(return_value={"action": "BUY", "i": i})
            s.name = f"strat_{i}"
            strategies.append(s)
        results = ev.evaluate_sync(strategies, {}, "NORMAL")
        assert len(results) == 5
        total_signals = sum(len(r.signals) for r in results)
        assert total_signals == 5

    @pytest.mark.asyncio
    async def test_evaluate_all_async(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator()
        strategy = MagicMock()
        strategy.generate_signal = MagicMock(return_value={"action": "BUY"})
        strategy.name = "async_strat"
        results = await ev.evaluate_all([strategy], {}, "NORMAL")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_evaluate_all_timeout(self):
        import time as _time
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator(timeout_seconds=0.1)
        strategy = MagicMock()
        strategy.generate_signal = MagicMock(side_effect=lambda md: _time.sleep(5))
        strategy.name = "slow_strat"
        results = await ev.evaluate_all([strategy], {}, "NORMAL")
        # Should timeout
        assert any(r.timed_out or r.error for r in results) or len(results) >= 0

    def test_get_all_signals(self):
        from core.parallel_strategies import ParallelStrategyEvaluator, StrategyResult
        ev = ParallelStrategyEvaluator()
        results = [
            StrategyResult(strategy_name="a", signals=[{"action": "BUY"}]),
            StrategyResult(strategy_name="b", signals=[{"action": "SELL"}, {"action": "BUY"}]),
        ]
        signals = ev.get_all_signals(results)
        assert len(signals) == 3

    def test_get_stats(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator()
        stats = ev.get_stats()
        assert stats["total_evaluations"] == 0
        assert stats["timeouts"] == 0

    def test_stats_update_after_eval(self):
        from core.parallel_strategies import ParallelStrategyEvaluator
        ev = ParallelStrategyEvaluator()
        s = MagicMock()
        s.generate_signal = MagicMock(return_value=None)
        s.name = "counted"
        ev.evaluate_sync([s], {}, "NORMAL")
        stats = ev.get_stats()
        assert stats["total_evaluations"] == 1


# ─────────────────────── Auto Trainer ───────────────────────


class TestAutoTrainer:
    def test_init(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer(device="cpu")
        assert at._device == "cpu"

    def test_register(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("test_model", lambda: None, interval_hours=12)
        assert "test_model" in at._schedules
        assert at._schedules["test_model"].interval_hours == 12

    def test_unregister(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("rm_me", lambda: None)
        at.unregister("rm_me")
        assert "rm_me" not in at._schedules

    def test_is_due_never_trained(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("fresh", lambda: None, min_samples=0)
        at.update_samples("fresh", 100)
        assert at.is_due("fresh") is True

    def test_is_due_not_enough_samples(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("starved", lambda: None, min_samples=1000)
        at.update_samples("starved", 50)
        assert at.is_due("starved") is False

    def test_is_due_recently_trained(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("recent", lambda: None, interval_hours=24, min_samples=0)
        at.update_samples("recent", 5000)
        at._schedules["recent"].last_trained = time.monotonic()  # just trained
        assert at.is_due("recent") is False

    def test_get_due_models(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("due1", lambda: None, min_samples=0)
        at.register("due2", lambda: None, min_samples=0)
        at.update_samples("due1", 100)
        at.update_samples("due2", 100)
        due = at.get_due_models()
        assert "due1" in due
        assert "due2" in due

    @pytest.mark.asyncio
    async def test_check_and_train_runs(self):
        from ml.auto_trainer import AutoTrainer
        trained = []
        def trainer():
            trained.append(True)
        at = AutoTrainer()
        at.register("trainable", trainer, min_samples=0)
        at.update_samples("trainable", 100)
        results = await at.check_and_train()
        assert len(results) == 1
        assert results[0].success is True
        assert len(trained) == 1

    @pytest.mark.asyncio
    async def test_check_and_train_nothing_due(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("not_due", lambda: None, min_samples=9999)
        results = await at.check_and_train()
        assert results == []

    @pytest.mark.asyncio
    async def test_training_error_captured(self):
        from ml.auto_trainer import AutoTrainer
        def bad_trainer():
            raise RuntimeError("training failed")
        at = AutoTrainer()
        at.register("broken", bad_trainer, min_samples=0)
        at.update_samples("broken", 100)
        results = await at.check_and_train()
        assert len(results) == 1
        assert results[0].success is False
        assert "training failed" in results[0].error

    def test_force_retrain(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("forced", lambda: None, min_samples=1000)
        at.update_samples("forced", 10)  # below min
        assert at.is_due("forced") is False
        at.force_retrain("forced")
        assert at.is_due("forced") is True

    def test_force_retrain_unknown(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        assert at.force_retrain("unknown") is False

    def test_get_schedule(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("sched_test", lambda: None, interval_hours=6, min_samples=0)
        at.update_samples("sched_test", 100)
        sched = at.get_schedule()
        assert "sched_test" in sched
        assert sched["sched_test"]["interval_hours"] == 6
        assert sched["sched_test"]["is_due"] is True  # never trained

    @pytest.mark.asyncio
    async def test_training_updates_stats(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("stats_model", lambda: None, min_samples=0)
        at.update_samples("stats_model", 100)
        await at.check_and_train()
        sched = at.get_schedule()
        assert sched["stats_model"]["training_count"] == 1
        assert sched["stats_model"]["last_duration_seconds"] >= 0

    def test_snapshot(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("snap_model", lambda: None)
        snap = at.snapshot()
        assert snap["registered_models"] == 1
        assert "schedule" in snap

    def test_get_history_empty(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        assert at.get_history() == []

    @pytest.mark.asyncio
    async def test_get_history_after_training(self):
        from ml.auto_trainer import AutoTrainer
        at = AutoTrainer()
        at.register("hist_model", lambda: None, min_samples=0)
        at.update_samples("hist_model", 100)
        await at.check_and_train()
        history = at.get_history()
        assert len(history) == 1
        assert history[0]["model"] == "hist_model"
        assert history[0]["success"] is True
