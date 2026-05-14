"""Small YAML/JSON config loader for adaptive modules."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path


@dataclass
class AdaptiveConfig:
    capital: float = 10000
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    max_drawdown: float = 0.15
    max_daily_loss: float = 0.05
    learning_interval_seconds: float = 0.5
    paper_mode: bool = True


class AdaptiveConfigManager:
    def __init__(self, path: str | Path = "adaptive_config.yaml"):
        self.path = Path(path)

    def load(self) -> AdaptiveConfig:
        if not self.path.exists():
            return self.from_env(AdaptiveConfig())
        text = self.path.read_text(encoding="utf-8")
        data = self._parse_yaml_like(text) if self.path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
        return self.from_env(self._config_from_mapping(data))

    def save_default(self) -> None:
        cfg = AdaptiveConfig()
        data = cfg.__dict__
        if self.path.suffix.lower() in {".yaml", ".yml"}:
            lines = []
            for key, value in data.items():
                if isinstance(value, list):
                    lines.append(f"{key}:")
                    lines.extend(f"  - {item}" for item in value)
                else:
                    lines.append(f"{key}: {json.dumps(value)}")
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _parse_yaml_like(text: str) -> dict[str, object]:
        data: dict[str, object] = {}
        current_list: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if line.startswith("  - ") and current_list:
                values = data.setdefault(current_list, [])
                if isinstance(values, list):
                    values.append(line[4:].strip())
                continue
            current_list = None
            if line.endswith(":"):
                current_list = line[:-1].strip()
                data[current_list] = []
                continue
            key, _, value = line.partition(":")
            value = value.strip()
            try:
                data[key.strip()] = json.loads(value)
            except json.JSONDecodeError:
                data[key.strip()] = value
        return data

    @staticmethod
    def _config_from_mapping(data: dict[str, object]) -> AdaptiveConfig:
        symbols_value = data.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        symbols = [str(item) for item in symbols_value] if isinstance(symbols_value, list) else [str(symbols_value)]
        return AdaptiveConfig(
            capital=AdaptiveConfigManager._as_float(data.get("capital"), 10000),
            symbols=symbols,
            max_drawdown=AdaptiveConfigManager._as_float(data.get("max_drawdown"), 0.15),
            max_daily_loss=AdaptiveConfigManager._as_float(data.get("max_daily_loss"), 0.05),
            learning_interval_seconds=AdaptiveConfigManager._as_float(data.get("learning_interval_seconds"), 0.5),
            paper_mode=bool(data.get("paper_mode", True)),
        )

    @staticmethod
    def _as_float(value: object, default: float) -> float:
        if isinstance(value, (int, float, str)):
            return float(value)
        return default

    @staticmethod
    def from_env(config: AdaptiveConfig) -> AdaptiveConfig:
        config.capital = float(os.getenv("ARGUS_INITIAL_EQUITY", config.capital))
        config.max_drawdown = float(os.getenv("ARGUS_MAX_DRAWDOWN", config.max_drawdown))
        config.max_daily_loss = float(os.getenv("ARGUS_MAX_DAILY_LOSS_PCT", config.max_daily_loss))
        config.paper_mode = os.getenv("ARGUS_PAPER_MODE", str(config.paper_mode)).lower() in {"1", "true", "yes"}
        symbols = os.getenv("ARGUS_SYMBOLS")
        if symbols:
            config.symbols = [symbol.strip() for symbol in symbols.split(",") if symbol.strip()]
        return config


def _demo() -> None:
    print("Adaptive config manager ready")
    print(AdaptiveConfigManager().load())


if __name__ == "__main__":
    _demo()
