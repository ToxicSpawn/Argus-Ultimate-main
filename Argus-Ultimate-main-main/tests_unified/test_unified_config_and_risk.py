from __future__ import annotations

import unittest


class TestUnifiedConfigAndRisk(unittest.TestCase):
    def test_strict_validation_rejects_unknown_nested_keys(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown risk key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02, "typoo_key": 1},
                }
            )

    def test_strict_validation_rejects_out_of_range_numeric(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "risk.max_daily_loss_pct"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 1.5},
                }
            )

    def test_unified_config_from_yaml_dict_basic_mapping(self) -> None:
        from unified_trading_system import UnifiedConfig

        y = {
            "capital": {"starting_capital_aud": 1234.0, "min_position_size_aud": 5.0, "max_position_size_aud": 40.0},
            "risk": {
                "max_daily_loss_pct": 0.01,
                "max_drawdown_pct": 0.10,
                "max_consecutive_losses": 2,
                "portfolio_var_limit_pct": 0.03,
                "portfolio_cvar_limit_pct": 0.05,
                "portfolio_var_confidence": 0.97,
                "portfolio_var_lookback_trades": 120,
                "cluster_drawdown_brake_pct": 0.06,
                "target_cluster_cap_pct": 0.35,
                "risk_cluster_map": {"BTC/USD": "majors"},
                "portfolio_vol_target_pct": 1.5,
                "portfolio_liquidity_spread_ref_bps": 25.0,
                "portfolio_exposure_min_scale": 0.4,
            },
            "execution": {"slippage_pct": 0.002},
            "exchanges": {
                "primary": "kraken",
                "secondary": "coinbase_advanced",
                "kraken": {"taker_fee": 0.0026},
                "coinbase_advanced": {"taker_fee": 0.008},
            },
            "monitoring": {
                "prometheus": {"enabled": True, "port": 9191},
                "grafana": {"enabled": False, "port": 3333},
            },
            "multi_language": {"enabled": False},
            "trading_pairs": ["BTC/USD", "ETH/USD"],
        }

        cfg = UnifiedConfig.from_unified_yaml_dict(y)
        self.assertEqual(cfg.starting_capital_aud, 1234.0)
        self.assertEqual(cfg.min_position_size_aud, 5.0)
        self.assertEqual(cfg.max_position_size_aud, 40.0)
        self.assertEqual(cfg.max_daily_loss_pct, 0.01)
        self.assertEqual(cfg.max_drawdown_pct, 0.10)
        self.assertEqual(cfg.max_consecutive_losses, 2)
        self.assertEqual(cfg.portfolio_var_limit_pct, 0.03)
        self.assertEqual(cfg.portfolio_cvar_limit_pct, 0.05)
        self.assertEqual(cfg.portfolio_var_confidence, 0.97)
        self.assertEqual(cfg.portfolio_var_lookback_trades, 120)
        self.assertEqual(cfg.cluster_drawdown_brake_pct, 0.06)
        self.assertEqual(cfg.target_cluster_cap_pct, 0.35)
        self.assertEqual(cfg.risk_cluster_map.get("BTC/USD"), "majors")
        self.assertEqual(cfg.portfolio_vol_target_pct, 1.5)
        self.assertEqual(cfg.portfolio_liquidity_spread_ref_bps, 25.0)
        self.assertEqual(cfg.portfolio_exposure_min_scale, 0.4)
        self.assertEqual(cfg.slippage_pct, 0.002)
        self.assertEqual(cfg.primary_exchange, "kraken")
        self.assertEqual(cfg.secondary_exchange, "coinbase_advanced")
        self.assertEqual(cfg.kraken_taker_fee, 0.0026)
        self.assertEqual(cfg.coinbase_taker_fee, 0.008)
        self.assertTrue(cfg.prometheus_enabled)
        self.assertEqual(cfg.prometheus_port, 9191)
        self.assertFalse(cfg.grafana_enabled)
        self.assertFalse(cfg.multi_language_enabled)
        self.assertEqual(cfg.trading_pairs, ["BTC/USD", "ETH/USD"])

    def test_unified_risk_manager_daily_loss_limit(self) -> None:
        from risk.unified_risk_manager import UnifiedRiskManager

        rm = UnifiedRiskManager(initial_capital=1000.0, max_daily_loss=0.02)
        rm.update_capital(1000.0, pnl=-25.0)  # -2.5%
        self.assertTrue(rm.is_daily_loss_limit_exceeded())

    def test_unified_risk_manager_circuit_breaker_on_consecutive_losses(self) -> None:
        from risk.unified_risk_manager import UnifiedRiskManager

        rm = UnifiedRiskManager(initial_capital=1000.0, max_consecutive_losses=2)
        rm.record_trade(-1.0)
        rm.record_trade(-1.0)
        self.assertTrue(rm.check_circuit_breaker())

    def test_strict_validation_rejects_strategy_evaluation_unknown_key(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown strategy_evaluation_engine key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "strategy_evaluation_engine": {"enabled": True, "unknown_knob": 1},
                }
            )

    def test_strict_validation_rejects_strategy_evaluation_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "strategy_evaluation_engine.persist_interval_cycles"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "strategy_evaluation_engine": {"enabled": True, "persist_interval_cycles": 0},
                }
            )

    def test_strict_validation_rejects_champion_challenger_unknown_key(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown champion_challenger key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "champion_challenger": {"enabled": True, "unknown_knob": 1},
                }
            )

    def test_strict_validation_rejects_champion_challenger_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "champion_challenger.min_trades_for_promotion"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "champion_challenger": {"enabled": True, "min_trades_for_promotion": 0},
                }
            )

    def test_strict_validation_rejects_liquidity_risk_unknown_key(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown liquidity_risk_engine key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "liquidity_risk_engine": {"enabled": True, "unknown_knob": 1},
                }
            )

    def test_strict_validation_rejects_liquidity_risk_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "liquidity_risk_engine.depth_fraction_limit"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "liquidity_risk_engine": {"enabled": True, "depth_fraction_limit": 1.5},
                }
            )

    def test_strict_validation_rejects_liquidity_min_score_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "liquidity_risk_engine.min_liquidity_score"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "liquidity_risk_engine": {"enabled": True, "min_liquidity_score": 1.2},
                }
            )

    def test_strict_validation_rejects_meta_engine_unknown_key(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown self_optimizing_meta_engine key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "self_optimizing_meta_engine": {"enabled": True, "unknown_knob": 1},
                }
            )

    def test_strict_validation_rejects_meta_engine_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "self_optimizing_meta_engine.meta_alpha"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "self_optimizing_meta_engine": {"enabled": True, "meta_alpha": 1.5},
                }
            )

    def test_strict_validation_rejects_microstructure_engine_unknown_key(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown market_microstructure_engine key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "market_microstructure_engine": {"enabled": True, "unknown_knob": 1},
                }
            )

    def test_strict_validation_rejects_microstructure_engine_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "market_microstructure_engine.vacuum_depth_drop_ratio"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "market_microstructure_engine": {"enabled": True, "vacuum_depth_drop_ratio": 1.5},
                }
            )

    def test_strict_validation_rejects_recon_recovery_unknown_key(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown recon_recovery_engine key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "recon_recovery_engine": {"enabled": True, "unknown_knob": 1},
                }
            )

    def test_strict_validation_rejects_recon_recovery_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "recon_recovery_engine.max_retries"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "recon_recovery_engine": {"enabled": True, "max_retries": 0},
                }
            )

    def test_strict_validation_rejects_recon_recovery_halt_flag_type(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "recon_recovery_engine.halt_on_retry_exhausted"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "recon_recovery_engine": {"enabled": True, "halt_on_retry_exhausted": "yes"},
                }
            )

    def test_strict_validation_rejects_system_health_metrics_unknown_key(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown system_health_metrics key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "system_health_metrics": {"enabled": True, "unknown_knob": 1},
                }
            )

    def test_strict_validation_rejects_system_health_metrics_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "system_health_metrics.snapshot_interval_cycles"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "system_health_metrics": {"enabled": True, "snapshot_interval_cycles": 0},
                }
            )

    def test_strict_validation_rejects_runtime_safety_unknown_key(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "unknown runtime_safety key"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "runtime_safety": {"latency_grace_cycles": 2, "unknown_knob": 1},
                }
            )

    def test_strict_validation_rejects_market_data_out_of_range(self) -> None:
        from core.config_manager import validate_unified_config_dict

        with self.assertRaisesRegex(ValueError, "market_data.ohlcv_retry_attempts"):
            validate_unified_config_dict(
                {
                    "config_version": 1,
                    "runtime": {"mode": "paper"},
                    "capital": {"starting_capital_aud": 1000.0},
                    "risk": {"max_daily_loss_pct": 0.02},
                    "market_data": {"ohlcv_retry_attempts": 99},
                }
            )
