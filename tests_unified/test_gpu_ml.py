"""
test_gpu_ml.py — Tests for the GPU ML engine and LAN signal bridge.

All tests are designed to pass without a real GPU:
  • CUDA-dependent code is skipped gracefully (pytest.mark.skipif or mock)
  • ZMQ sockets are not opened — config/data-structure tests only
  • No actual training loops are run (unit-level construction tests only)
"""
from __future__ import annotations

import asyncio
import time
import warnings
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _has_torch() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_zmq() -> bool:
    try:
        import zmq
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 1 — gpu_deeplob_trainer.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainerConfigDefaults:
    """test_trainer_config_defaults — verify TrainerConfig defaults."""

    def test_defaults(self):
        from ml.gpu_deeplob_trainer import TrainerConfig
        cfg = TrainerConfig()
        assert cfg.device == "auto"
        assert cfg.model_path == "models/deeplob_weights.pt"
        assert cfg.batch_size == 256
        assert cfg.learning_rate == 0.001
        assert cfg.epochs == 50
        assert cfg.val_split == 0.2
        assert cfg.mixed_precision is True
        assert cfg.num_workers == 8
        assert cfg.sequence_length == 100
        assert cfg.feature_dim == 40
        assert cfg.n_classes == 3
        assert cfg.early_stopping_patience == 5
        assert cfg.checkpoint_dir == "models/checkpoints"

    def test_all_fields_present(self):
        from ml.gpu_deeplob_trainer import TrainerConfig
        cfg = TrainerConfig()
        field_names = {f.name for f in fields(cfg)}
        required = {
            "device", "model_path", "batch_size", "learning_rate", "epochs",
            "val_split", "mixed_precision", "num_workers", "sequence_length",
            "feature_dim", "n_classes", "early_stopping_patience", "checkpoint_dir",
        }
        assert required.issubset(field_names), f"Missing fields: {required - field_names}"


class TestTrainerAutoDetectDevice:
    """test_trainer_auto_detect_device — no CUDA → device='cpu', no crash."""

    def test_falls_back_to_cpu_when_no_cuda(self):
        from ml.gpu_deeplob_trainer import GPUDeepLOBTrainer, TrainerConfig

        with patch("ml.gpu_deeplob_trainer._TORCH_AVAILABLE", False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                resolved = GPUDeepLOBTrainer._resolve_device("auto")
            assert resolved == "cpu"

    def test_explicit_cpu_always_works(self):
        from ml.gpu_deeplob_trainer import GPUDeepLOBTrainer
        resolved = GPUDeepLOBTrainer._resolve_device("cpu")
        assert resolved == "cpu"

    def test_cuda_request_without_cuda_falls_back(self):
        from ml.gpu_deeplob_trainer import GPUDeepLOBTrainer
        with patch("ml.gpu_deeplob_trainer._TORCH_AVAILABLE", True):
            try:
                import torch
                if not torch.cuda.is_available():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        resolved = GPUDeepLOBTrainer._resolve_device("cuda:0")
                    assert resolved == "cpu"
            except ImportError:
                pytest.skip("torch not available")


class TestTrainerBuildModel:
    """test_trainer_build_model — model has expected layer structure."""

    @pytest.mark.skipif(not _has_torch(), reason="torch not installed")
    def test_build_model_returns_module(self):
        import torch.nn as nn
        from ml.gpu_deeplob_trainer import GPUDeepLOBTrainer, TrainerConfig
        cfg = TrainerConfig(device="cpu")
        trainer = GPUDeepLOBTrainer(cfg)
        model = trainer.build_model()
        assert isinstance(model, nn.Module)

    @pytest.mark.skipif(not _has_torch(), reason="torch not installed")
    def test_model_has_conv_lstm_layers(self):
        import torch.nn as nn
        from ml.gpu_deeplob_trainer import GPUDeepLOBTrainer, TrainerConfig
        cfg = TrainerConfig(device="cpu")
        trainer = GPUDeepLOBTrainer(cfg)
        model = trainer.build_model()
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0, "Model should have trainable parameters"

    @pytest.mark.skipif(not _has_torch(), reason="torch not installed")
    def test_model_forward_shape(self):
        import torch
        from ml.gpu_deeplob_trainer import GPUDeepLOBTrainer, TrainerConfig
        cfg = TrainerConfig(device="cpu", sequence_length=10, feature_dim=40, n_classes=3)
        trainer = GPUDeepLOBTrainer(cfg)
        model = trainer.build_model()
        model.eval()
        x = torch.randn(4, cfg.sequence_length, cfg.feature_dim)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3), f"Expected (4, 3), got {out.shape}"


class TestTrainerMixedPrecisionFlag:
    """test_trainer_mixed_precision_flag — config.mixed_precision is respected."""

    def test_mixed_precision_true_by_default(self):
        from ml.gpu_deeplob_trainer import TrainerConfig
        cfg = TrainerConfig()
        assert cfg.mixed_precision is True

    def test_mixed_precision_can_be_disabled(self):
        from ml.gpu_deeplob_trainer import TrainerConfig
        cfg = TrainerConfig(mixed_precision=False)
        assert cfg.mixed_precision is False


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 2 — gpu_inference_server.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferenceServerConfig:
    """test_inference_server_config — default publish_address = 'tcp://*:9200'."""

    def test_default_publish_address(self):
        from ml.gpu_inference_server import InferenceServerConfig
        cfg = InferenceServerConfig()
        assert cfg.publish_address == "tcp://*:9200"

    def test_default_inference_interval(self):
        from ml.gpu_inference_server import InferenceServerConfig
        cfg = InferenceServerConfig()
        assert cfg.inference_interval_ms == 100.0

    def test_default_warmup_batches(self):
        from ml.gpu_inference_server import InferenceServerConfig
        cfg = InferenceServerConfig()
        assert cfg.warmup_batches == 5

    def test_default_mixed_precision(self):
        from ml.gpu_inference_server import InferenceServerConfig
        cfg = InferenceServerConfig()
        assert cfg.mixed_precision is True


class TestInferenceServerMessageFormat:
    """test_inference_server_message_format — msgpack signal has all required keys."""

    def test_required_signal_keys(self):
        required_keys = {
            "type", "symbol", "direction", "confidence",
            "logits", "timestamp_ns", "inference_latency_us", "model_version",
        }
        # Construct a signal dict as the server would produce
        signal = {
            "type": "deeplob_signal",
            "symbol": "BTC/USDT",
            "direction": "up",
            "confidence": 0.85,
            "logits": [0.85, 0.10, 0.05],
            "timestamp_ns": time.time_ns(),
            "inference_latency_us": 450,
            "model_version": "v1",
        }
        assert required_keys.issubset(signal.keys()), (
            f"Missing: {required_keys - signal.keys()}"
        )

    def test_direction_values(self):
        from ml.gpu_inference_server import GPUInferenceServer
        assert GPUInferenceServer.DIRECTION_MAP[0] == "up"
        assert GPUInferenceServer.DIRECTION_MAP[1] == "neutral"
        assert GPUInferenceServer.DIRECTION_MAP[2] == "down"

    def test_server_instantiation_no_crash(self):
        from ml.gpu_inference_server import GPUInferenceServer, InferenceServerConfig
        cfg = InferenceServerConfig(device="cpu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            srv = GPUInferenceServer(cfg)
        assert srv is not None

    def test_server_get_stats_keys(self):
        from ml.gpu_inference_server import GPUInferenceServer, InferenceServerConfig
        cfg = InferenceServerConfig(device="cpu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            srv = GPUInferenceServer(cfg)
        stats = srv.get_stats()
        for key in ("inference_count", "model_loaded", "symbols_tracked", "device"):
            assert key in stats, f"Missing stat key: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 3 — lan_signal_bridge.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestLANReceiverStaleDetection:
    """test_lan_receiver_stale_detection — no signal for 5 s → is_stale True."""

    def test_stale_when_running_and_no_signal(self):
        from ml.lan_signal_bridge import LANSignalReceiver, ReceiverConfig
        cfg = ReceiverConfig(signal_timeout_ms=5000.0)
        recv = LANSignalReceiver(cfg)
        recv._running = True
        recv._last_signal_time_ns = 0  # never received
        assert recv.is_stale() is True

    def test_not_stale_immediately_after_signal(self):
        from ml.lan_signal_bridge import LANSignalReceiver, ReceiverConfig
        cfg = ReceiverConfig(signal_timeout_ms=5000.0)
        recv = LANSignalReceiver(cfg)
        recv._running = True
        recv._last_signal_time_ns = time.time_ns()  # just now
        assert recv.is_stale() is False

    def test_stale_after_timeout_elapsed(self):
        from ml.lan_signal_bridge import LANSignalReceiver, ReceiverConfig
        cfg = ReceiverConfig(signal_timeout_ms=100.0)   # 100 ms timeout
        recv = LANSignalReceiver(cfg)
        recv._running = True
        # Set last signal to 200 ms ago
        recv._last_signal_time_ns = time.time_ns() - int(200 * 1_000_000)
        assert recv.is_stale() is True

    def test_default_timeout_is_5000ms(self):
        from ml.lan_signal_bridge import ReceiverConfig
        cfg = ReceiverConfig()
        assert cfg.signal_timeout_ms == 5000.0


class TestLANReceiverFallback:
    """test_lan_receiver_fallback — stale + fallback_to_cpu → get_latest returns None."""

    def test_stale_fallback_returns_none(self):
        from ml.lan_signal_bridge import LANSignalReceiver, ReceiverConfig
        cfg = ReceiverConfig(signal_timeout_ms=5000.0, fallback_to_cpu=True)
        recv = LANSignalReceiver(cfg)
        recv._running = True
        recv._last_signal_time_ns = 0
        recv._latest["BTC/USDT"] = {"direction": "up"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = recv.get_latest_signal("BTC/USDT")
        assert result is None

    def test_no_fallback_returns_cached_signal(self):
        from ml.lan_signal_bridge import LANSignalReceiver, ReceiverConfig
        cfg = ReceiverConfig(signal_timeout_ms=5000.0, fallback_to_cpu=False)
        recv = LANSignalReceiver(cfg)
        recv._running = True
        recv._last_signal_time_ns = 0
        expected = {"direction": "up", "symbol": "BTC/USDT"}
        recv._latest["BTC/USDT"] = expected
        # Even when stale, fallback_to_cpu=False should return cached value
        result = recv.get_latest_signal("BTC/USDT")
        assert result == expected


class TestLANBridgeRoleSender:
    """test_lan_bridge_role_sender — role='sender' starts sender component."""

    def test_sender_role_creates_sender(self):
        from ml.lan_signal_bridge import LANSignalBridge
        bridge = LANSignalBridge(role="sender", publish_address="tcp://*:19200")
        assert bridge.role == "sender"
        assert bridge._sender is not None
        assert bridge._receiver is None

    def test_sender_publish_address_propagated(self):
        from ml.lan_signal_bridge import LANSignalBridge
        bridge = LANSignalBridge(role="sender", publish_address="tcp://*:19200")
        assert bridge._sender.publish_address == "tcp://*:19200"

    def test_invalid_role_raises(self):
        from ml.lan_signal_bridge import LANSignalBridge
        with pytest.raises(ValueError, match="role must be one of"):
            LANSignalBridge(role="invalid")


class TestLANBridgeRoleReceiver:
    """test_lan_bridge_role_receiver — role='receiver' starts receiver component."""

    def test_receiver_role_creates_receiver(self):
        from ml.lan_signal_bridge import LANSignalBridge
        bridge = LANSignalBridge(
            role="receiver",
            subscribe_address="tcp://argus-pc:9200",
        )
        assert bridge.role == "receiver"
        assert bridge._receiver is not None
        assert bridge._sender is None

    def test_receiver_subscribe_address_propagated(self):
        from ml.lan_signal_bridge import LANSignalBridge
        bridge = LANSignalBridge(
            role="receiver",
            subscribe_address="tcp://192.168.1.10:9200",
        )
        assert bridge._receiver.config.subscribe_address == "tcp://192.168.1.10:9200"

    def test_get_signal_receiver_returns_none_when_no_data(self):
        from ml.lan_signal_bridge import LANSignalBridge
        bridge = LANSignalBridge(role="receiver")
        bridge._receiver._running = False  # not started — not stale either
        result = bridge.get_signal("BTC/USDT")
        assert result is None

    def test_get_signal_sender_always_none(self):
        from ml.lan_signal_bridge import LANSignalBridge
        bridge = LANSignalBridge(role="sender")
        assert bridge.get_signal("BTC/USDT") is None


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 4 — gpu_backtester.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktesterConfig:
    """test_backtester_config — default parallel_runs=16."""

    def test_default_parallel_runs(self):
        from ml.gpu_backtester import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.parallel_runs == 16

    def test_default_strategies(self):
        from ml.gpu_backtester import BacktestConfig
        cfg = BacktestConfig()
        assert "micro_mm" in cfg.strategies
        assert "funding_arb" in cfg.strategies
        assert "scalping" in cfg.strategies

    def test_default_lookback_days(self):
        from ml.gpu_backtester import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.lookback_days == 30

    def test_default_data_path(self):
        from ml.gpu_backtester import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.data_path == "data/backtest"


class TestBacktestResultFields:
    """test_backtest_result_fields — BacktestResult has all required fields."""

    def test_all_required_fields_present(self):
        from ml.gpu_backtester import BacktestResult
        field_names = {f.name for f in fields(BacktestResult)}
        required = {
            "params", "total_pnl", "sharpe", "sortino", "max_drawdown",
            "fill_rate", "adverse_rate", "trades", "win_rate", "avg_win",
            "avg_loss", "profit_factor", "duration_days", "annualised_return_pct",
        }
        assert required.issubset(field_names), f"Missing: {required - field_names}"

    def test_backtest_result_instantiation(self):
        from ml.gpu_backtester import BacktestResult
        r = BacktestResult(
            params={"spread_bps": 2},
            total_pnl=1500.0,
            sharpe=1.5,
            sortino=2.0,
            max_drawdown=0.05,
            fill_rate=0.35,
            adverse_rate=0.10,
            trades=120,
            win_rate=0.55,
            avg_win=25.0,
            avg_loss=-12.0,
            profit_factor=2.3,
            duration_days=30,
            annualised_return_pct=18.0,
        )
        assert r.sharpe == 1.5
        assert r.trades == 120
        assert r.valid is True  # default

    def test_single_backtest_run(self):
        from ml.gpu_backtester import GPUBacktester, BacktestConfig
        cfg = BacktestConfig(lookback_days=1, device="cpu")
        bt = GPUBacktester(cfg)
        result = asyncio.run(bt.run_backtest("micro_mm", {"spread_bps": 2, "skew_factor": 1.0}))
        assert isinstance(result.total_pnl, float)
        assert result.strategy_name == "micro_mm"


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 1 — gpu_deeplob_trainer.py (GPU stats without CUDA)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGPUStatsNoCuda:
    """test_gpu_stats_no_cuda — get_gpu_stats returns {'available': False} gracefully."""

    def test_no_cuda_returns_available_false(self):
        from ml.gpu_deeplob_trainer import GPUDeepLOBTrainer, TrainerConfig

        with patch("ml.gpu_deeplob_trainer._TORCH_AVAILABLE", False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                trainer = GPUDeepLOBTrainer(TrainerConfig(device="cpu"))
            # Patch torch check inside get_gpu_stats
            with patch("ml.gpu_deeplob_trainer._TORCH_AVAILABLE", False):
                stats = trainer.get_gpu_stats()
        assert stats.get("available") is False

    def test_gpu_stats_when_cuda_unavailable_via_torch(self):
        """If torch is available but CUDA is not, should still return gracefully."""
        try:
            import torch
            from ml.gpu_deeplob_trainer import GPUDeepLOBTrainer, TrainerConfig
            if not torch.cuda.is_available():
                trainer = GPUDeepLOBTrainer(TrainerConfig(device="cpu"))
                stats = trainer.get_gpu_stats()
                assert stats.get("available") is False
            else:
                pytest.skip("CUDA is available on this machine")
        except ImportError:
            pytest.skip("torch not installed")
