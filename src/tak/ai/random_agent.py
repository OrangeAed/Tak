"""The easiest opponent: a random legal move every turn."""

from __future__ import annotations

import random

from tak.ai.base import Agent
from tak.engine.game import Game
from tak.engine.moves import Move


class RandomAgent(Agent):
    """Plays a legal move picked at random, every move equally likely."""

    key = "random"
    name = "Random"
    description = "plays a random legal move"

    def pick(self, game: Game, rng: random.Random) -> Move:
        return rng.choice(game.legal_moves())
