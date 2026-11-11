"""The GUI app: the size argument and turning typed PTN into moves."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tak.engine import Game
from tak.gui.app import BAD, GOOD, main, submit_move


class TestSubmitMove:
    def test_legal_move_plays_quietly(self) -> None:
        game = Game(5)
        assert submit_move(game, "a1") is None
        assert game.ply == 1

    def test_bad_notation_is_reported(self) -> None:
        game = Game(5)
        kind, message = submit_move(game, "xyzzy")
        assert kind == BAD and message
        assert game.ply == 0

    def test_illegal_move_is_reported(self) -> None:
        game = Game(5)
        submit_move(game, "a1")
        kind, message = submit_move(game, "a1")
        assert kind == BAD and "Illegal move" in message
        assert game.ply == 1

    def test_threatening_move_announces_tak(self, started: Callable[..., Game]) -> None:
        game = started("a1 c3 b1 b3", size=3)
        assert submit_move(game, "b2") == (GOOD, "Tak!")


class TestMain:
    def test_rejects_bad_size(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--size", "9"])
        assert "--size" in capsys.readouterr().err

    def test_size_goes_to_the_game(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sizes: list[int] = []
        monkeypatch.setattr("tak.gui.app.run", lambda game: sizes.append(game.size))
        assert main(["--size", "6"]) == 0
        assert sizes == [6]
