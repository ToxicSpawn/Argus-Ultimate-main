"""
Run Bayesian Hyperparameter Optimization for Argus-Ultimate v5.0.0
Calibrated for $1k capital accounts.

Usage:
    python scripts/run_bayesian_opt.py [--trials 100] [--study-name argus_hpt_1k]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.bayesian_optimizer import BayesianOptimizer, SmallCapObjective, DEFAULT_PARAM_SPACE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_bayesian_opt")


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian HPO for Argus-Ultimate $1k capital")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--startup-trials", type=int, default=15, help="Random startup trials before TPE")
    parser.add_argument("--study-name", type=str, default="argus_hpt_1k", help="Optuna study name")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital in USD")
    parser.add_argument("--results-dir", type=str, default="optimization/results", help="Output directory")
    parser.add_argument("--no-pruning", action="store_true", help="Disable MedianPruner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"=== Argus-Ultimate Bayesian HPO ===")
    logger.info(f"Capital: ${args.capital:.2f} | Trials: {args.trials} | Study: {args.study_name}")

    # Build objective
    objective = SmallCapObjective(
        capital=args.capital,
        roi_weight=0.5,
        sharpe_weight=0.3,
        drawdown_weight=0.2,
    )

    # Build optimizer
    optimizer = BayesianOptimizer(
        objective_fn=objective,
        param_space=DEFAULT_PARAM_SPACE,
        n_trials=args.trials,
        n_startup_trials=args.startup_trials,
        study_name=args.study_name,
        direction="maximize",
        results_dir=args.results_dir,
        seed=args.seed,
        pruning=not args.no_pruning,
    )

    # Run
    study = optimizer.run()

    # Report
    best = study.best_trial
    logger.info(f"\n{'='*50}")
    logger.info(f"Optimization complete.")
    logger.info(f"Best trial: #{best.number} | Score: {best.value:.6f}")
    logger.info(f"Best params:")
    for k, v in best.params.items():
        logger.info(f"  {k}: {v}")

    # Parameter importances
    importances = optimizer.importance()
    if importances:
        logger.info(f"\nParameter importances:")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            logger.info(f"  {k}: {v:.4f}")

    logger.info(f"Results saved to: {args.results_dir}/{args.study_name}_results.json")


if __name__ == "__main__":
    main()
