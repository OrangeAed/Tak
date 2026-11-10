"""Shared fixtures for the threat tests: spotting "Tak" and "Tinue"."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator

import pytest

from tak.cli.app import run
from tak.engine import Color, Game, GameResult, Move, Piece, PieceType, Reserve, Square

_LABELS = {
    "w": Piece(Color.WHITE, PieceType.FLAT),
    "b": Piece(Color.BLACK, PieceType.FLAT),
    "Sw": Piece(Color.WHITE, PieceType.WALL),
    "Sb": Piece(Color.BLACK, PieceType.WALL),
    "Cw": Piece(Color.WHITE, PieceType.CAP),
    "Cb": Piece(Color.BLACK, PieceType.CAP),
}

Position = Callable[..., Game]
WinsWhenPlayed = Callable[[Game, Color, Move], bool]
RandomPositions = Callable[[int, int, int], Iterator[Game]]


@pytest.fixture
def position() -> Position:
    """Set up a game directly: ``position(3, {"c1": "b w w"}, white=(1, 0))``.

    Stacks are written bottom first with the board's labels (``w``, ``Sb``,
    ``Cw``...). Reserves are full unless given as ``(stones, capstones)``.
    No moves have been played, so it is nominally White's turn; the threat
    checks don't depend on whose turn it is.
    """

    def make(size: int, stacks: dict[str, str], **reserves: tuple[int, int]) -> Game:
        game = Game(size)
        for name, labels in stacks.items():
            pieces = tuple(_LABELS[label] for label in labels.split())
            game.board.set_stack(Square.from_name(name), pieces)
        for color in Color:
            if color.value in reserves:
                game.reserves[color] = Reserve(*reserves[color.value])
        return game

    return make


@pytest.fixture
def wins_when_played() -> WinsWhenPlayed:
    """Whether ``move`` wins for ``color`` (who must be to move) when really played."""

    def check(game: Game, color: Color, move: Move) -> bool:
        assert color is game.to_move
        game.play(move)
        try:
            return game.result is not None and game.result.winner is color
        finally:
            game.undo()

    return check


@pytest.fixture
def random_positions() -> RandomPositions:
    """``random_positions(size, seed, max_plies)``: a random game, yielded after each move."""

    def generate(size: int, seed: int, max_plies: int) -> Iterator[Game]:
        rng = random.Random(seed * 7919 + size)
        game = Game(size)
        while not game.is_over and game.ply < max_plies:
            game.play(rng.choice(game.legal_moves()))
            yield game

    return generate


class Session:
    """Feeds scripted lines to the terminal game loop and records what it printed."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.output: list[str] = []

    def read(self, prompt: str) -> str:
        try:
            return next(self._lines)
        except StopIteration:
            raise EOFError from None

    def write(self, text: str) -> None:
        self.output.append(text)

    @property
    def lines(self) -> list[str]:
        """Everything printed, split into single lines."""
        return "\n".join(self.output).splitlines()


PlayScript = Callable[..., tuple[GameResult | None, list[str]]]


@pytest.fixture
def play_script() -> PlayScript:
    """Play scripted input through the CLI: returns the result and the printed lines."""

    def play(lines: list[str], size: int = 3) -> tuple[GameResult | None, list[str]]:
        session = Session(lines)
        result = run(Game(size), session.read, session.write)
        return result, session.lines

    return play
