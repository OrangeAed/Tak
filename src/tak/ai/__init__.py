"""Computer players (agents) to play Tak against.

Every agent implements :class:`Agent`: given a game, it returns a move. There
are three, from easiest to hardest:

- :class:`RandomAgent` (``random``) plays a random legal move.
- :class:`GreedyAgent` (``greedy``) takes an immediate win, blocks the
  opponent's immediate win, and otherwise favors moves that gain squares.
- :class:`StrategistAgent` (``strategist``) also plays for Tinue, and
  otherwise picks the move that most improves its position (roads, the
  center, squares controlled).

All three follow fixed, hand-written rules; none of them learns.
"""

from __future__ import annotations

from tak.ai.base import Agent
from tak.ai.greedy_agent import GreedyAgent
from tak.ai.random_agent import RandomAgent
from tak.ai.strategist_agent import StrategistAgent

#: Every agent by its short name, easiest first.
AGENTS: dict[str, type[Agent]] = {
    agent.key: agent for agent in (RandomAgent, GreedyAgent, StrategistAgent)
}


def make_agent(key: str, seed: int | None = None) -> Agent:
    """A new agent of the kind named ``key`` (``random``, ``greedy`` or ``strategist``)."""
    try:
        kind = AGENTS[key]
    except KeyError:
        choices = ", ".join(AGENTS)
        raise ValueError(f"unknown computer player {key!r}; choose one of {choices}") from None
    return kind(seed)


__all__ = ["AGENTS", "Agent", "GreedyAgent", "RandomAgent", "StrategistAgent", "make_agent"]
