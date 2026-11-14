"""Shared setup for the GUI tests: run pygame headless, skip if it's missing."""

from __future__ import annotations

import os
import random
import threading

import pytest

from tak.ai import Agent
from tak.engine import Game, Move, parse_move

pygame = pytest.importorskip("pygame")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="session", autouse=True)
def _pygame() -> object:
    pygame.init()
    yield pygame
    pygame.quit()


class Scripted(Agent):
    """A computer player that plays the PTN moves it is given, in order.

    Set ``gate`` to a :class:`threading.Event` to make it wait before each move
    until the event is set, to test what happens while the computer is thinking.
    """

    key = "scripted"
    name = "Scripty"
    description = "plays the moves it is given"

    def __init__(self, *moves: str, gate: threading.Event | None = None) -> None:
        super().__init__()
        self.moves = list(moves)
        self.gate = gate
        self.calls = 0

    def pick(self, game: Game, rng: random.Random) -> Move:
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if not self.moves:
            raise RuntimeError("out of moves")
        return parse_move(self.moves.pop(0))
