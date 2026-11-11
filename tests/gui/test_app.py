"""The GUI app: the size argument and turning typed PTN into moves."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tak.engine import Game, parse_ptn
from tak.gui.app import BAD, GOOD, INFO, GameRecorder, main, submit_move

# White completes a road from b1 up to c3 with c2.
ROAD_WIN = ["a1", "c3", "b1", "b3", "b2", "a3", "c2"]


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
        monkeypatch.setattr("tak.gui.app.run", lambda game, save_dir: sizes.append(game.size))
        assert main(["--size", "6"]) == 0
        assert sizes == [6]

    def test_save_options_reach_the_window(
        self, monkeypatch: pytest.MonkeyPatch, saves_dir: Path, tmp_path: Path
    ) -> None:
        folders: list[Path | None] = []
        monkeypatch.setattr("tak.gui.app.run", lambda game, save_dir: folders.append(save_dir))
        main([])
        main(["--saves-dir", str(tmp_path)])
        main(["--no-save"])
        assert folders == [saves_dir, tmp_path, None]


class TestGameRecorder:
    def play(self, game: Game, recorder: GameRecorder, moves: list[str]) -> tuple[str, str] | None:
        shown = None
        for move in moves:
            feedback = submit_move(game, move)
            assert feedback is None or feedback[0] != BAD, feedback  # every move is legal
            shown = recorder.after_move(game)
        return shown

    def test_saves_when_the_game_ends(self, tmp_path: Path) -> None:
        game, recorder = Game(3), GameRecorder(tmp_path)
        assert self.play(game, recorder, ROAD_WIN[:-1]) is None  # nothing saved mid-game
        assert list(tmp_path.iterdir()) == []
        kind, message = self.play(game, recorder, ROAD_WIN[-1:]) or ("", "")
        [path] = list(tmp_path.iterdir())
        assert kind == INFO and "Game saved to" in message and path.name in message
        assert parse_ptn(path.read_text(encoding="utf-8")).result == "R-0"

    def test_finishing_again_after_undo_updates_the_same_file(self, tmp_path: Path) -> None:
        game, recorder = Game(3), GameRecorder(tmp_path)
        self.play(game, recorder, ROAD_WIN)
        first = recorder.path
        game.undo()
        self.play(game, recorder, ["c2"])
        assert recorder.path == first
        assert list(tmp_path.iterdir()) == [first]

    def test_a_new_game_gets_a_new_file(self, tmp_path: Path) -> None:
        for _ in range(2):
            self.play(Game(3), GameRecorder(tmp_path), ROAD_WIN)
        assert len(list(tmp_path.iterdir())) == 2

    def test_no_save_dir_means_no_saving(self, saves_dir: Path) -> None:
        game, recorder = Game(3), GameRecorder(None)
        assert self.play(game, recorder, ROAD_WIN) is None
        assert game.is_over
        assert list(saves_dir.iterdir()) == []

    def test_a_failed_save_is_shown_as_an_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("", encoding="utf-8")
        kind, message = self.play(Game(3), GameRecorder(blocker), ROAD_WIN) or ("", "")
        assert kind == BAD and "Could not save the game" in message
