"""The window's game session: stepping through moves, loading, confirming, saving."""

from __future__ import annotations

from pathlib import Path

import pytest

from tak.engine import Color, Game, Square, game_to_ptn, parse_move, parse_ptn
from tak.gui.session import BAD, GOOD, INFO, GameRecorder, GameSession

from .conftest import Scripted

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


class TestAgainstTheComputer:
    def session(self, **players: Scripted) -> GameSession:
        return GameSession(
            Game(3), players={Color(color): agent for color, agent in players.items()}
        )

    def test_whose_turn(self) -> None:
        session = self.session(black=Scripted("c3"))
        assert session.computer_to_move is None and not session.waiting_for_computer
        session.play_text("a1")
        assert session.computer_to_move is session.players[Color.BLACK]
        assert session.waiting_for_computer

    def test_which_computer_made_the_last_move(self) -> None:
        session = self.session(black=Scripted())
        assert session.computer_that_moved is None  # no moves yet
        session.play_text("a1")
        assert session.computer_that_moved is None  # yours
        session.play_computer(parse_move("c3"))
        assert session.computer_that_moved is session.players[Color.BLACK]
        session.back()
        assert session.computer_that_moved is None
        assert GameSession(Game(3)).computer_that_moved is None

    def test_who_plays_whom(self) -> None:
        assert self.session(black=Scripted()).versus_text() == "You play White against Scripty"
        assert self.session(white=Scripted()).versus_text() == "You play Black against Scripty"
        assert GameSession(Game(3)).versus_text() is None

    def test_a_person_cant_move_for_the_computer(self) -> None:
        session = self.session(white=Scripted())
        for attempt in (lambda: session.play(parse_move("a1")), lambda: session.play_text("a1")):
            kind, text = attempt()  # type: ignore[misc]
            assert kind == BAD and "It's Scripty's turn (White)" in text
        assert session.game.ply == 0

    def test_the_computers_moves_are_announced(self) -> None:
        session = self.session(black=Scripted())
        session.play_text("a1")
        assert session.play_computer(parse_move("c3")) == (
            INFO,
            "Scripty placed a white flat on c3 for you.",
        )
        session.play_text("b2")
        assert session.play_computer(parse_move("b3")) == (INFO, "Scripty played b3.")

    def test_a_threat_by_the_computer_is_called(self) -> None:
        session = self.session(black=Scripted())
        session.play_text("a2")  # a black flat for the computer
        session.play_computer(parse_move("c3"))
        session.play_text("a1")
        assert session.play_computer(parse_move("b2")) == (GOOD, "Scripty played b2. Tak!")

    def test_only_on_the_computers_turn_at_the_latest_position(self) -> None:
        session = self.session(black=Scripted())
        with pytest.raises(ValueError, match="isn't the computer's turn"):
            session.play_computer(parse_move("a1"))
        session.play_text("a1")
        session.future.append(parse_move("c3"))  # its turn, but with a move already ahead
        assert session.computer_to_move is not None and not session.waiting_for_computer
        with pytest.raises(ValueError):
            session.play_computer(parse_move("c3"))

    def test_stepping_is_one_move_at_a_time_as_in_any_game(self) -> None:
        session = self.session(black=Scripted())
        session.play_text("a1")
        session.play_computer(parse_move("c3"))
        session.play_text("b2")
        session.play_computer(parse_move("b3"))
        assert session.back() and session.game.ply == 3
        assert session.back() and session.game.ply == 2
        assert session.forward() and session.game.ply == 3
        session.to_start()
        assert session.game.ply == 0
        session.to_end()
        assert session.game.ply == 4 and not session.forward()

    def test_the_computer_waits_at_an_earlier_turn_of_its_own(self) -> None:
        session = self.session(black=Scripted())
        session.play_text("a1")
        session.play_computer(parse_move("c3"))
        session.back()
        assert session.computer_to_move is not None and not session.waiting_for_computer
        kind, text = session.position_note()  # type: ignore[misc]
        assert kind == INFO
        assert text == (
            "Scripty (Black) moves next. Step forward to see its move, "
            "or back to your turn to play something different."
        )
        kind, text = session.play_text("b3")  # type: ignore[misc]
        assert kind == BAD and "Step forward to see its move" in text
        assert session.game.ply == 1

    def test_no_note_elsewhere(self) -> None:
        session = self.session(black=Scripted())
        assert session.position_note() is None  # your turn
        session.play_text("a1")
        assert session.position_note() is None  # the computer's turn, at the end: it moves
        assert played("a1", "c3").position_note() is None  # no computer

    def test_playing_black_the_computers_opening_can_be_stepped_back_over(self) -> None:
        session = self.session(white=Scripted())
        session.play_computer(parse_move("a1"))
        assert session.back() and session.game.ply == 0
        assert session.position_note() is not None
        assert session.forward() and session.game.ply == 1

    def test_playing_on_from_earlier_lets_the_computer_answer(self) -> None:
        session = self.session(black=Scripted())
        session.play_text("a1")
        session.play_computer(parse_move("c3"))
        session.play_text("b2")
        session.play_computer(parse_move("b3"))
        session.back()
        session.back()
        assert session.play_text("b1") is None  # replaces b2 and b3, as in any game
        assert session.total == 3 and session.waiting_for_computer

    def test_a_game_won_by_the_computer_is_saved_with_the_players(self, tmp_path: Path) -> None:
        session = GameSession(Game(3), tmp_path, {Color.BLACK: Scripted()})
        session.play_text("a2")
        session.play_computer(parse_move("c3"))
        session.play_text("a1")
        session.play_computer(parse_move("b2"))
        session.play_text("b1")
        kind, text = session.play_computer(parse_move("c2"))
        assert kind == INFO and text.startswith("Scripty played c2. Game saved to ")
        text = next(tmp_path.glob("*.ptn")).read_text(encoding="utf-8")
        assert '[Player1 "Human"]' in text and '[Player2 "Scripty (computer)"]' in text
