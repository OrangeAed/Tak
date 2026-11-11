"""The window's game session: stepping through moves, loading, confirming, saving."""

from __future__ import annotations

from pathlib import Path

import pytest

from tak.engine import Game, Square, game_to_ptn, parse_move, parse_ptn
from tak.gui.session import BAD, GOOD, INFO, GameRecorder, GameSession

# White completes a road from b1 up to c3 with c2.
ROAD_WIN = ["a1", "c3", "b1", "b3", "b2", "a3", "c2"]


def played(*moves: str, size: int = 3) -> GameSession:
    session = GameSession(Game(size))
    for move in moves:
        before = session.game.ply
        session.play_text(move)
        assert session.game.ply == before + 1, move  # every scripted move is legal
    return session


def saved_game(folder: Path, moves: list[str], name: str = "game.ptn") -> Path:
    game = Game(3)
    for move in moves:
        game.play(parse_move(move))
    path = folder / name
    path.write_text(game_to_ptn(game), encoding="utf-8")
    return path


class TestStepping:
    def test_back_and_forward(self) -> None:
        session = played("a1", "c3", "b1")
        assert session.back() and session.back()
        assert session.game.ply == 1 and session.total == 3
        assert session.forward()
        assert session.game.ply == 2
        assert session.game.board.stack(Square.from_name("c3"))

    def test_stops_at_the_ends(self) -> None:
        session = played("a1")
        assert not session.forward()
        assert session.back()
        assert not session.back()

    def test_jump_to_start_and_end(self) -> None:
        session = played(*ROAD_WIN)
        session.to_start()
        assert session.game.ply == 0 and not session.game.is_over
        session.to_end()
        assert session.game.ply == len(ROAD_WIN) and session.game.is_over

    def test_position_text(self) -> None:
        session = played("a1", "c3")
        assert session.position_text() == "Move 2 of 2: c3"
        session.back()
        assert session.position_text() == "Move 1 of 2: a1"
        session.back()
        assert session.position_text() == "Move 0 of 2"

    def test_a_new_move_part_way_replaces_the_rest_of_your_own_game(self) -> None:
        session = played("a1", "c3", "b1")
        session.back()
        session.back()
        assert session.play_text("a3") is None  # no confirmation for your own game
        assert session.total == 2 and session.future == []


class TestLoading:
    def test_opens_at_the_start(self, tmp_path: Path) -> None:
        session = GameSession.load(saved_game(tmp_path, ROAD_WIN))
        assert session.game.ply == 0
        assert session.total == len(ROAD_WIN)
        session.to_end()
        assert session.game.is_over

    def test_stepping_through_a_finished_game_does_not_save_it_again(
        self, tmp_path: Path, saves_dir: Path
    ) -> None:
        session = GameSession.load(saved_game(tmp_path, ROAD_WIN), saves_dir)
        session.to_end()
        session.to_start()
        session.to_end()
        assert list(saves_dir.iterdir()) == []

    def test_playing_part_way_asks_first(self, tmp_path: Path) -> None:
        session = GameSession.load(saved_game(tmp_path, ROAD_WIN))
        session.forward()
        kind, message = session.play_text("b2") or ("", "")
        assert kind == INFO and "replaces the 6 later moves" in message and "Enter" in message
        assert session.game.ply == 1  # nothing played yet
        assert session.pending == parse_move("b2")

    def test_confirming_plays_it_and_drops_the_rest(self, tmp_path: Path) -> None:
        session = GameSession.load(saved_game(tmp_path, ROAD_WIN))
        session.forward()
        session.play_text("b2")
        session.confirm()
        assert session.game.ply == 2 and session.total == 2
        assert session.source is None and session.pending is None
        session.play_text("a2")  # no more asking: it's a new game now
        assert session.game.ply == 3 and session.pending is None

    def test_playing_the_same_move_again_also_confirms(self, tmp_path: Path) -> None:
        session = GameSession.load(saved_game(tmp_path, ROAD_WIN))
        session.play_text("b2")
        session.play_text("b2")
        assert session.game.ply == 1 and session.total == 1

    def test_cancelling_and_stepping_forget_the_pending_move(self, tmp_path: Path) -> None:
        session = GameSession.load(saved_game(tmp_path, ROAD_WIN))
        session.play_text("b2")
        assert session.cancel_pending() is True
        assert session.confirm() is None and session.game.ply == 0
        session.play_text("b2")
        session.forward()
        assert session.pending is None

    def test_an_illegal_move_is_rejected_without_asking(self, tmp_path: Path) -> None:
        session = GameSession.load(saved_game(tmp_path, ROAD_WIN))
        session.forward()
        kind, _ = session.play_text("a1") or ("", "")  # occupied
        assert kind == BAD and session.pending is None

    def test_playing_on_from_the_end_needs_no_confirmation(self, tmp_path: Path) -> None:
        session = GameSession.load(saved_game(tmp_path, ROAD_WIN[:4]))
        session.to_end()
        session.play_text("b2")
        assert session.game.ply == 5

    def test_a_different_ending_is_saved_as_a_new_file(
        self, tmp_path: Path, saves_dir: Path
    ) -> None:
        path = saved_game(tmp_path, ROAD_WIN)
        original = path.read_text(encoding="utf-8")
        session = GameSession.load(path, saves_dir)
        session.to_end()
        session.back()  # back to before White's winning c2
        session.play_text("c1")
        session.confirm()  # White plays c1 instead, which doesn't win...
        session.play_text("a2")  # ...and Black completes the a1-a2-a3 road
        assert session.game.is_over
        [saved] = list(saves_dir.iterdir())
        assert parse_ptn(saved.read_text(encoding="utf-8")).result == "0-R"
        assert path.read_text(encoding="utf-8") == original  # the loaded file is untouched

    def test_missing_and_broken_files(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            GameSession.load(tmp_path / "nope.ptn")
        broken = tmp_path / "broken.ptn"
        broken.write_text('[Size "5"]\n1. a1 a1\n', encoding="utf-8")
        with pytest.raises(ValueError):
            GameSession.load(broken)


class TestFeedback:
    def test_tak_is_announced(self) -> None:
        session = played("a1", "c3", "b1", "b3")
        assert session.play_text("b2") == (GOOD, "Tak!")

    def test_bad_notation(self) -> None:
        kind, _ = GameSession(Game(5)).play_text("xyzzy") or ("", "")
        assert kind == BAD


class TestGameRecorder:
    def test_saves_when_the_game_ends(self, tmp_path: Path) -> None:
        session = GameSession(Game(3), tmp_path)
        for move in ROAD_WIN[:-1]:
            session.play_text(move)
        assert list(tmp_path.iterdir()) == []  # nothing saved mid-game
        kind, message = session.play_text(ROAD_WIN[-1]) or ("", "")
        [path] = list(tmp_path.iterdir())
        assert kind == INFO and "Game saved to" in message and path.name in message

    def test_finishing_again_after_stepping_back_updates_the_same_file(
        self, tmp_path: Path
    ) -> None:
        session = GameSession(Game(3), tmp_path)
        for move in ROAD_WIN:
            session.play_text(move)
        first = session.recorder.path
        session.back()
        session.play_text("c2")
        assert session.recorder.path == first
        assert list(tmp_path.iterdir()) == [first]

    def test_no_save_dir_means_no_saving(self, saves_dir: Path) -> None:
        session = GameSession(Game(3), None)
        for move in ROAD_WIN:
            session.play_text(move)
        assert session.game.is_over and list(saves_dir.iterdir()) == []

    def test_a_failed_save_is_shown_as_an_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("", encoding="utf-8")
        recorder = GameRecorder(blocker)
        session = played(*ROAD_WIN)
        kind, message = recorder.after_move(session.game) or ("", "")
        assert kind == BAD and "Could not save the game" in message
