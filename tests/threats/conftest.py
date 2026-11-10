"""Shared helpers for the tests of threat detection ("Tak" and "Tinue").

These are fixtures that return functions, like the ones in ``tests/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from tak.cli.app import run
from tak.engine import (
    BOARD_CONFIG,
    Color,
    Game,
    GameResult,
    Move,
    Piece,
    PieceType,
    PlaceMove,
    Reserve,
    Square,
)

Position = Callable[..., Game]
WinsWhenPlayed = Callable[[Game, Color, Move], bool]
Transcript = tuple[Game, GameResult | None, list[str]]
PlayScript = Callable[..., Transcript]

LABELS = {
    "w": Piece(Color.WHITE, PieceType.FLAT),
    "b": Piece(Color.BLACK, PieceType.FLAT),
    "Sw": Piece(Color.WHITE, PieceType.WALL),
    "Sb": Piece(Color.BLACK, PieceType.WALL),
    "Cw": Piece(Color.WHITE, PieceType.CAP),
    "Cb": Piece(Color.BLACK, PieceType.CAP),
}

_SWAPPED = {"w": "b", "b": "w", "Sw": "Sb", "Sb": "Sw", "Cw": "Cb", "Cb": "Cw"}


@pytest.fixture
def position() -> Position:
    """Set up a game directly: ``position(3, {"c1": "b w w"})``.

    Stacks are written bottom first, using the board's labels (``w``, ``Sb``,
    ``Cw``, ...). Reserves start full; override them with e.g. ``white=(1, 0)``
    for one stone and no capstones. The game is past the opening, with
    ``to_move`` (White unless given) to play, so its moves can really be played.
    """

    def make(
        size: int,
        stacks: dict[str, str],
        to_move: Color = Color.WHITE,
        **reserves: tuple[int, int],
    ) -> Game:
        game = Game(size)
        # A couple of throwaway placements get past the opening with the right
        # player to move; then the board is cleared and laid out as asked.
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


@pytest.fixture
def swap_colors() -> Callable[[dict[str, str]], dict[str, str]]:
    """Turn a ``position`` layout into the same layout with the colors swapped."""

    def swap(stacks: dict[str, str]) -> dict[str, str]:
        return {
            name: " ".join(_SWAPPED[label] for label in labels.split())
            for name, labels in stacks.items()
        }

    return swap


@pytest.fixture
def wins_when_played() -> WinsWhenPlayed:
    """Whether ``move`` wins for ``color`` (who must be to move), using the real engine."""

    def check(game: Game, color: Color, move: Move) -> bool:
        assert color is game.to_move
        game.play(move)
        try:
            return game.result is not None and game.result.winner is color
        finally:
            game.undo()

    return check


class Session:
    """Feeds scripted lines to the game loop and records what it printed."""

    def __init__(self, lines: list[str]) -> None:
        self._lines: Iterator[str] = iter(lines)
        self.output: list[str] = []

    def read(self, _prompt: str) -> str:
        try:
            return next(self._lines)
        except StopIteration:
            raise EOFError from None

    def write(self, text: str) -> None:
        self.output.append(text)


@pytest.fixture
def play_script() -> PlayScript:
    """Run the terminal game on scripted input.

    Returns the game, its result, and every printed line (multi-line writes
    are split, so each status line can be checked on its own).
    """

    def play(lines: list[str], size: int = 3) -> Transcript:
        game = Game(size)
        session = Session(lines)
        result = run(game, session.read, session.write)
        return game, result, "\n".join(session.output).splitlines()

    return play
