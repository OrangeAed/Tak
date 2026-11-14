"""The interface every computer player implements: given a game, return a move."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from tak.engine.board import Square
from tak.engine.game import Game
from tak.engine.moves import Move, PlaceMove
from tak.engine.pieces import Color


class Agent(ABC):
    """A computer player.

    Call :meth:`choose_move` with the game; it returns a legal move for the
    player whose turn it is and never changes the game it is given (it thinks
    on a copy). Every agent is ordinary code following fixed rules, with no
    learning of any kind. Where they make random choices (the random player's
    moves, the greedy player's weighted picks, breaking ties), ``self.rng`` is
    used, so passing a ``seed`` makes them repeatable.
    """

    #: The short name used to pick this player, e.g. on the command line.
    key: ClassVar[str]
    #: The name shown to the person playing against it.
    name: ClassVar[str]
    #: How it plays, in a few words.
    description: ClassVar[str]

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, game: Game) -> Move:
        """A legal move for the player to move. Raises :class:`ValueError` if there is none."""
        if game.is_over:
            raise ValueError("the game is over")
        return self.pick(game.copy())

    @abstractmethod
    def pick(self, game: Game) -> Move:
        """Choose a move. ``game`` is the agent's own copy, so it may play and undo moves on it."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def opening_move(game: Game, rng: random.Random) -> PlaceMove:
    """The first turn: put the opponent's flat stone in a corner.

    On the first turn each player places a stone for the *opponent*, so the
    usual choice is the least useful square there is for it: a corner, which
    has the fewest neighbours to build a road from.
    """
    last = game.size - 1
    corners = [Square(0, 0), Square(last, 0), Square(0, last), Square(last, last)]
    free = [square for square in corners if not game.board.stack(square)]
    return PlaceMove(rng.choice(free))


def player_headers(players: Mapping[Color, Agent]) -> dict[str, str]:
    """PTN ``Player1`` (White) and ``Player2`` (Black) headers naming who played.

    Sides in ``players`` are the computer's; the others are a person's. With
    no computer players there is nothing worth recording, so this is empty.
    """
    if not players:
        return {}
    names = {
        color: f"{players[color].name} (computer)" if color in players else "Human"
        for color in Color
    }
    return {"Player1": names[Color.WHITE], "Player2": names[Color.BLACK]}
