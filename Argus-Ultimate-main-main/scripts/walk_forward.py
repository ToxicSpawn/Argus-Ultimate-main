"""Walk-forward validation for time-series trading models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Iterable, Protocol
import random


class PredictiveModel(Protocol):
    def fit(self, features: Iterable[Iterable[float]], labels: Iterable[int]) -> object: ...
    def predict_proba(self, features: Iterable[Iterable[float]]) -> Iterable[Iterable[float]]: ...


@dataclass
class WalkForwardFold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    accuracy: float
    average_return: float
    sharpe: float


class WalkForwardValidator:
    def __init__(self, train_size: int = 120, test_size: int = 30, step_size: int = 30):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, n_rows: int):
        start = 0
        while start + self.train_size + self.test_size <= n_rows:
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            yield start, train_end, train_end, test_end
            start += self.step_size

    def validate(
        self,
        features: Iterable[Iterable[float]],
        labels: Iterable[int],
        returns: Iterable[float],
        model_factory: Callable[[], PredictiveModel],
    ) -> list[WalkForwardFold]:
        x = [list(row) for row in features]
        y = [int(label) for label in labels]
        r = [float(value) for value in returns]
        folds: list[WalkForwardFold] = []
        for train_start, train_end, test_start, test_end in self.split(len(x)):
            model = model_factory()
            model.fit(x[train_start:train_end], y[train_start:train_end])
            probs = list(model.predict_proba(x[test_start:test_end]))
            prob_rows = [list(row) for row in probs]
            pred = [max(range(len(row)), key=lambda idx: float(row[idx])) for row in prob_rows]
            direction = [1 if value == 2 else -1 if value == 0 else 0 for value in pred]
            strategy_returns = [side * ret for side, ret in zip(direction, r[test_start:test_end])]
            vol = self._std(strategy_returns)
            folds.append(WalkForwardFold(
                train_start,
                train_end,
                test_start,
                test_end,
                self._mean([float(a == b) for a, b in zip(pred, y[test_start:test_end])]),
                self._mean(strategy_returns),
                self._mean(strategy_returns) / (vol + 1e-9) * (252 ** 0.5),
            ))
        return folds

    @staticmethod
    def summary(folds: list[WalkForwardFold]) -> dict[str, float]:
        if not folds:
            return {"folds": 0, "accuracy": 0.0, "average_return": 0.0, "sharpe": 0.0}
        return {
            "folds": float(len(folds)),
            "accuracy": WalkForwardValidator._mean([f.accuracy for f in folds]),
            "average_return": WalkForwardValidator._mean([f.average_return for f in folds]),
            "sharpe": WalkForwardValidator._mean([f.sharpe for f in folds]),
        }

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _std(values: list[float]) -> float:
        if not values:
            return 0.0
        avg = WalkForwardValidator._mean(values)
        return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def _demo() -> None:
    try:
        LSTMLearner = getattr(import_module("scripts.lstm_learning"), "LSTMLearner")
    except ModuleNotFoundError:
        LSTMLearner = getattr(import_module("lstm_learning"), "LSTMLearner")

    rng = random.Random(11)
    x = [[rng.gauss(0, 0.02) for _ in range(9)] for _ in range(260)]
    future = [row[0] + 0.5 * row[1] for row in x]
    y = [2 if value > 0.01 else 0 if value < -0.01 else 1 for value in future]
    validator = WalkForwardValidator(train_size=100, test_size=40, step_size=40)
    folds = validator.validate(x, y, future, lambda: LSTMLearner(input_size=9, lookback=12))
    print("Walk-forward validation ready")
    print(WalkForwardValidator.summary(folds))


if __name__ == "__main__":
    _demo()
