"""
Planning algorithms for ARGUS.

- ``mcts``: Monte Carlo Tree Search with UCB1, for multi-step trade planning
  using the learned world model as the generative simulator.
"""

from .mcts import MCTS, MCTSNode, MCTSConfig

__all__ = ["MCTS", "MCTSNode", "MCTSConfig"]
