"""Shared helpers for the computer player tests."""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from tak.ai import make_agent
from tak.engine import (
    BOARD_CONFIG,
    Color,
    Game,
    Piece,
    PieceType,
    PlaceMove,
    Reserve,
    SlideMove,
    Square,
)

Position = Callable[..., Game]

LABELS = {
    "w": Piece(Color.WHITE, PieceType.FLAT),
    "b": Piece(Color.BLACK, PieceType.FLAT),
    "Sw": Piece(Color.WHITE, PieceType.WALL),
    "Sb": Piece(Color.BLACK, PieceType.WALL),
    "Cw": Piece(Color.WHITE, PieceType.CAP),
    "Cb": Piece(Color.BLACK, PieceType.CAP),
}


@pytest.fixture
def position() -> Position:
    """Set up a game directly: ``position(4, {"a1": "b w", "b2": "Cw"}, to_move=Color.BLACK)``.

    Stacks are written bottom first. Reserves start full; override them with
    e.g. ``white=(1, 0)``. The game is past the opening, with ``to_move``
    (White unless given) to play.
    """

    def make(
        size: int,
        stacks: dict[str, str],
        to_move: Color = Color.WHITE,
        **reserves: tuple[int, int],
    ) -> Game:
        game = Game(size)
        squares = list(game.board.squares())
        for square in squares[: 2 if to_move is Color.WHITE else 3]:
            game.play(PlaceMove(square))
        for square in squares:
            game.board.set_stack(square, ())
        for name, labels in stacks.items():
            pieces = tuple(LABELS[label] for label in labels.split())
            game.board.set_stack(Square.from_name(name), pieces)
        for color in Color:
            stones, caps = reserves.get(color.value, BOARD_CONFIG[size])
            game.reserves[color] = Reserve(stones, caps)
        assert game.to_move is to_move and not game.in_opening and not game.is_over
        return game

    return make


def random_positions(size: int, count: int, seed: int, slide_bias: float = 0.5) -> list[Game]:
    """Positions from random games, favoring slides so stacks build up."""
    rng = random.Random(seed)
    found: list[Game] = []
    while len(found) < count:
        game = Game(size)
        while not game.is_over and len(found) < count:
            moves = game.legal_moves()
            slides = [move for move in moves if isinstance(move, SlideMove)]
            pool = slides if slides and rng.random() < slide_bias else moves
            game.play(rng.choice(pool))
            if not game.is_over and not game.in_opening and rng.random() < 0.3:
                found.append(game.copy())
    return found


def bot_game_positions(size: int, seed: int, white: str, black: str) -> list[Game]:
    """Every position (past the opening) of a game between two computer players.

    These are full of threats, blocks and Tinue, unlike random positions.
    """
    agents = {Color.WHITE: make_agent(white, seed), Color.BLACK: make_agent(black, seed + 1)}
    game = Game(size)
    found = []
    while not game.is_over and game.ply < 200:
        if not game.in_opening:
            found.append(game.copy())
        game.play(agents[game.to_move].choose_move(game))
    return found
