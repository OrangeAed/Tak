"""Shared fixtures for the Tak tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tak.engine import Game, parse_move
from tak.saves import SAVES_ENV

Snapshot = tuple[Any, ...]


@pytest.fixture(autouse=True)
def saves_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where finished games are saved during a test: a fresh temporary folder.

    Applied to every test, so no test (nor any program a test starts, since the
    environment is inherited) can write into the project's real saves folder.
    """
    directory = tmp_path_factory.mktemp("saves")
    monkeypatch.setenv(SAVES_ENV, str(directory))
    return directory


@pytest.fixture
def play() -> Callable[[Game, str], None]:
    """Play space-separated PTN moves (no move numbers) on a game, in order."""

    def play_moves(game: Game, moves: str) -> None:
        for token in moves.split():
            game.play(parse_move(token))

    return play_moves


@pytest.fixture
def started(play: Callable[[Game, str], None]) -> Callable[..., Game]:
    """Build a game and play some PTN moves on it: ``started("a1 e5", size=5)``."""

    def make(moves: str = "", size: int = 5) -> Game:
        game = Game(size)
        play(game, moves)
        return game

    return make


@pytest.fixture
def snapshot() -> Callable[[Game], Snapshot]:
    """Capture everything about a game's position so two states can be compared."""

    def take(game: Game) -> Snapshot:
        stacks = tuple(game.board.stack(square) for square in game.board.squares())
        return (stacks, dict(game.reserves), game.result, game.ply)

    return take
