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
    on a copy). Agents don't learn while they play: even the learner, whose
    evaluation was trained beforehand, plays with fixed weights.

    **The same position always gets the same move.** Agents do make random
    choices (the random player's moves, the greedy player's weighted picks,
    breaking ties), but the randomness comes from the agent's :attr:`seed`
    together with the position (see :func:`position_key`), not from a
    generator that moves on with every call. So taking moves back and playing
    them again, or reaching a position by another order of moves, brings the
    same reply. Without a ``seed``, each agent picks its own at random, so a
    new agent (a new game) can play differently; the same ``seed`` plays the
    same in every game.
    """

    #: The short name used to pick this player, e.g. on the command line.
    key: ClassVar[str]
    #: The name shown to the person playing against it.
    name: ClassVar[str]
    #: How it plays, in a few words.
    description: ClassVar[str]

    def __init__(self, seed: int | None = None) -> None:
        #: Mixed into every random choice; fixed for the agent's lifetime.
        self.seed = seed if seed is not None else random.randrange(2**63)

    def choose_move(self, game: Game) -> Move:
        """A legal move for the player to move. Raises :class:`ValueError` if there is none.

        For a given agent, this depends only on the position.
        """
        if game.is_over:
            raise ValueError("the game is over")
        return self.pick(game.copy(), self.rng_for(game))

    def rng_for(self, game: Game) -> random.Random:
        """The random numbers :meth:`pick` gets for ``game``'s position."""
        # A string seed is hashed the same way in every run (unlike hash()).
        return random.Random(f"{self.seed}|{position_key(game)}")

    @abstractmethod
    def pick(self, game: Game, rng: random.Random) -> Move:
        """Choose a move.

        ``game`` is the agent's own copy, so it may play and undo moves on it.
        ``rng`` is for any random choices: it's seeded from the position, so
        using it (and nothing else random) keeps the choice the same whenever
        the position comes up again.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def position_key(game: Game) -> str:
    """A text description of everything that decides what a player can do next.

    Two games in the same position (the same stacks, reserves, and player to
    move) get the same key, however they got there.
    """
    stacks = "/".join(
        "".join(f"{piece.type.value}{piece.color.value[0]}" for piece in game.board.stack(square))
        for square in game.board.squares()
    )
    reserves = " ".join(f"{game.reserve(c).stones},{game.reserve(c).caps}" for c in Color)
    opening = "opening" if game.in_opening else "regular"
    return f"{game.size}x{game.size} {game.to_move.value} {opening} {reserves} {stacks}"


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
