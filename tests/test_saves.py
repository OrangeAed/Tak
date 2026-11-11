"""Saving finished games as PTN files."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

from tak import saves
from tak.engine import Color, Game, GameResult, Reason, game_from_ptn, parse_ptn

WHEN = datetime(2026, 11, 9, 14, 3, 22)

# White completes a road from b1 up to c3 with c2.
ROAD_WIN = "a1 c3 b1 b3 b2 a3 c2"


@pytest.fixture
def finished(started: Callable[..., Game]) -> Game:
    return started(ROAD_WIN, size=3)


class TestSaveGame:
    def test_writes_a_new_ptn_file_named_for_the_game(self, finished: Game, tmp_path: Path) -> None:
        path = saves.save_game(finished, tmp_path, when=WHEN)
        assert path.parent == tmp_path
        assert path.name == "2026-11-09_14-03-22_3x3_white-road.ptn"
        assert list(tmp_path.iterdir()) == [path]

    def test_the_file_loads_back_as_the_same_game(self, finished: Game, tmp_path: Path) -> None:
        path = saves.save_game(finished, tmp_path, when=WHEN)
        loaded = game_from_ptn(path.read_text(encoding="utf-8"))
        assert [played.move for played in loaded.history] == [
            played.move for played in finished.history
        ]
        assert loaded.result == finished.result == GameResult(Color.WHITE, Reason.ROAD)

    def test_records_size_result_and_when_it_was_played(
        self, finished: Game, tmp_path: Path
    ) -> None:
        path = saves.save_game(finished, tmp_path, when=WHEN)
        headers = parse_ptn(path.read_text(encoding="utf-8")).headers
        assert headers == {"Size": "3", "Date": "2026.11.09", "Time": "14:03:22", "Result": "R-0"}

    def test_never_overwrites_an_existing_save(self, finished: Game, tmp_path: Path) -> None:
        first = saves.save_game(finished, tmp_path, when=WHEN)
        original = first.read_text(encoding="utf-8")
        second = saves.save_game(finished, tmp_path, when=WHEN)
        third = saves.save_game(finished, tmp_path, when=WHEN)
        assert second.name == "2026-11-09_14-03-22_3x3_white-road-2.ptn"
        assert third.name == "2026-11-09_14-03-22_3x3_white-road-3.ptn"
        assert first.read_text(encoding="utf-8") == original
        assert len(list(tmp_path.iterdir())) == 3

    def test_creates_the_folder_if_needed(self, finished: Game, tmp_path: Path) -> None:
        folder = tmp_path / "new" / "saves"
        path = saves.save_game(finished, folder, when=WHEN)
        assert path.parent == folder and path.is_file()

    def test_replace_rewrites_an_earlier_save(self, finished: Game, tmp_path: Path) -> None:
        path = saves.save_game(finished, tmp_path, when=WHEN)
        finished.undo()  # the game is no longer over
        assert saves.save_game(finished, tmp_path, when=WHEN, replace=path) == path
        assert list(tmp_path.iterdir()) == [path]  # rewritten, not a second file
        assert parse_ptn(path.read_text(encoding="utf-8")).result is None

    def test_uses_the_default_folder_when_none_is_given(
        self, finished: Game, saves_dir: Path
    ) -> None:
        path = saves.save_game(finished, when=WHEN)
        assert path.parent == saves_dir


class TestOutcomeSlug:
    def test_road_win(self, finished: Game) -> None:
        assert saves.outcome_slug(finished) == "white-road"

    def test_flat_win_and_draw(self) -> None:
        game = Game(3)
        game.result = GameResult(Color.BLACK, Reason.FLAT)
        assert saves.outcome_slug(game) == "black-flats"
        game.result = GameResult(None, Reason.FLAT)
        assert saves.outcome_slug(game) == "draw"

    def test_unfinished(self) -> None:
        assert saves.outcome_slug(Game(3)) == "unfinished"


class TestWhereSavesGo:
    def test_environment_variable_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(saves.SAVES_ENV, "/somewhere/else")
        assert saves.default_saves_dir() == Path("/somewhere/else")

    def test_project_saves_folder_when_running_from_a_checkout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(saves.SAVES_ENV)
        project = Path(__file__).resolve().parents[1]
        assert saves.default_saves_dir() == project / "saves"

    def parse(self, *argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        saves.add_save_arguments(parser)
        return parser.parse_args(argv)

    def test_options(self, saves_dir: Path, tmp_path: Path) -> None:
        assert saves.saves_dir_from(self.parse()) == saves_dir
        assert saves.saves_dir_from(self.parse("--saves-dir", str(tmp_path))) == tmp_path
        assert saves.saves_dir_from(self.parse("--no-save")) is None

    def test_the_options_exclude_each_other(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            self.parse("--no-save", "--saves-dir", str(tmp_path))


class TestSaveFinishedGame:
    def test_reports_where_the_game_went(self, finished: Game, tmp_path: Path) -> None:
        path, message = saves.save_finished_game(finished, tmp_path)
        assert path is not None and path.is_file()
        assert message.startswith("Game saved to ") and path.name in message

    def test_a_failed_save_is_reported_not_raised(self, finished: Game, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("", encoding="utf-8")
        path, message = saves.save_finished_game(finished, blocker)
        assert path is None
        assert message.startswith("Could not save the game")
