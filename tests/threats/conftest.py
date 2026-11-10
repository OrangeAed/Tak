"""Shared helpers for the tests of threat detection ("Tak" and "Tinue").

These are fixtures that return functions, like the ones in ``tests/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from tak.cli.app import run
from tak.engine import Color, Game, GameResult, Move, Piece, PieceType, Reserve, Square

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
    for one stone and no capstones. No moves have been played.
    """

    def make(size: int, stacks: dict[str, str], **reserves: tuple[int, int]) -> Game:
        game = Game(size)
        for name, labels in stacks.items():
            pieces = tuple(LABELS[label] for label in labels.split())
            game.board.set_stack(Square.from_name(name), pieces)
        for color in Color:
            if color.value in reserves:
                game.reserves[color] = Reserve(*reserves[color.value])
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
