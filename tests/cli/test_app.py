"""Tests for the terminal game loop, driven by scripted input."""

from __future__ import annotations

import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from tak.ai import Agent, make_agent
from tak.cli.app import main, run
from tak.engine import (
    Color,
    Game,
    GameResult,
    Move,
    Piece,
    PieceType,
    Reason,
    Square,
    game_from_ptn,
    parse_move,
)

ROAD_GAME = ["a1", "a2", "b2", "a3", "c2"]  # on a 3x3 board: White wins with a2-b2-c2


class Session:
    """Feeds scripted lines to the game loop and records what it printed."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.prompts: list[str] = []
        self.output: list[str] = []

    def read(self, prompt: str) -> str:
        self.prompts.append(prompt)
        try:
            return next(self._lines)
        except StopIteration:
            raise EOFError from None

    def write(self, text: str) -> None:
        self.output.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.output)


def play_script(lines: list[str], size: int = 3) -> tuple[Game, GameResult | None, Session]:
    game = Game(size)
    session = Session(lines)
    result = run(game, session.read, session.write)
    return game, result, session


def test_a_game_runs_to_a_road_win() -> None:
    game, result, session = play_script(ROAD_GAME)
    assert result == GameResult(Color.WHITE, Reason.ROAD)
    assert game.result == result
    assert "Game over, White wins with a road: a2 -> b2 -> c2" in session.text
    assert "Result: R-0" in session.text


def test_the_loop_stops_asking_once_someone_wins() -> None:
    _, _, session = play_script([*ROAD_GAME, "a1", "a1"])
    assert len(session.prompts) == 5  # the extra lines are never read


def test_prompts_name_the_player_and_explain_the_opening() -> None:
    _, _, session = play_script(ROAD_GAME)
    assert "White, place a black flat stone" in session.prompts[0]
    assert "Black, place a white flat stone" in session.prompts[1]
    assert session.prompts[2] == "White > "
    assert session.prompts[3] == "Black > "


def test_each_turn_shows_the_board_and_status() -> None:
    _, _, session = play_script(ROAD_GAME)
    assert session.text.count("Turn ") == 5
    assert "White: 10 stones" in session.text


def test_illegal_moves_are_explained_and_the_same_player_tries_again() -> None:
    game, result, session = play_script(["a1", "a1", "a2", "quit"])
    assert "Illegal move: a1 is already occupied." in session.text
    assert result is None
    assert game.ply == 2  # a1 and a2; the repeated a1 was not played
    assert game.to_move is Color.WHITE
    assert session.prompts[1] == session.prompts[2]  # Black was asked again


def test_unreadable_input_gets_a_helpful_message() -> None:
    game, _, session = play_script(["banana", "quit"])
    assert "can't read 'banana' as a move" in session.text
    assert game.ply == 0


def test_rule_breaking_slides_are_rejected_with_the_reason() -> None:
    game, _, session = play_script(["a1", "c3", "a1>", "quit"])
    assert "Illegal move: you don't control the stack on a1" in session.text
    assert game.ply == 2


def test_blank_lines_are_ignored() -> None:
    game, _, _ = play_script(["", "   ", "a1", "quit"])
    assert game.ply == 1


def test_undo_takes_back_the_last_move() -> None:
    game, _, session = play_script(["a1", "undo", "a2", "quit"])
    assert "Took back a1." in session.text
    assert game.ply == 1
    assert game.board.stack(Square.from_name("a2")) == (Piece(Color.BLACK, PieceType.FLAT),)
    assert game.board.stack(Square.from_name("a1")) == ()


def test_undo_with_nothing_played() -> None:
    _, _, session = play_script(["undo", "quit"])
    assert "There is nothing to undo." in session.text


def test_help_ptn_and_moves_commands() -> None:
    _, _, session = play_script(["help", "a1", "ptn", "moves", "quit"])
    assert "Enter moves in PTN" in session.text
    assert '[Size "3"]' in session.text
    assert "1. a1" in session.text
    # After the first placement, Black may put a white flat on any of the other eight squares.
    assert "b1 c1 a2 b2 c2 a3 b3 c3" in session.output


def test_quit_abandons_the_game() -> None:
    game, result, session = play_script(["a1", "quit"])
    assert result is None
    assert not game.is_over
    assert "Game abandoned." in session.text


def test_running_out_of_input_ends_cleanly() -> None:
    game, result, session = play_script(["a1"])
    assert result is None
    assert game.ply == 1
    assert "Input ended" in session.text


def test_a_flat_win_is_reported() -> None:
    script = ["a2", "a3", "c3", "Sb1", "Sb2", "b3", "a1", "c2", "c1"]
    _, result, session = play_script(script)
    assert result == GameResult(Color.WHITE, Reason.FLAT)
    assert "White wins on flat stones, 4 to 3" in session.text
    assert "Result: F-0" in session.text


def test_a_longer_game_with_stacks_and_a_capstone() -> None:
    script = ["a1", "e5", "Cc3", "d3", "b5", "Sd4", "b4", "d4-", "c3>", "quit"]
    game, _, session = play_script(script, size=5)
    assert game.ply == 9
    assert "Illegal" not in session.text
    assert "Cw" in session.text  # the capstone shows on the board


def test_ptn_command_marks_flattening() -> None:
    script = ["a1", "e5", "Cc3", "d3", "b5", "Sd4", "b4", "d4-", "c3>", "ptn", "quit"]
    _, _, session = play_script(script, size=5)
    assert "5. c3>*" in session.text


class TestMain:
    def test_main_plays_a_full_game(self) -> None:
        session = Session(ROAD_GAME)
        assert main(["--size", "3"], session.read, session.write) == 0
        assert "Tak on a 3x3 board" in session.text
        assert "Result: R-0" in session.text

    def test_the_evaluation_bar(self) -> None:
        session = Session(ROAD_GAME)
        assert main(["--size", "3", "--eval-bar"], session.read, session.write) == 0
        assert " |w|" in session.text and "Evaluation: It's about even" in session.text
        boards = [text for text in session.output if "|w|" in text or "|b|" in text]
        assert boards[-1].splitlines()[-1].startswith(" R-0 ")  # the final board: White won
        plain = Session(ROAD_GAME)
        main(["--size", "3"], plain.read, plain.write)
        assert "|w|" not in plain.text and "Evaluation" not in plain.text

    def test_default_board_is_five_by_five(self) -> None:
        session = Session(["quit"])
        main([], session.read, session.write)
        assert "Tak on a 5x5 board" in session.text

    @pytest.mark.parametrize("size", ["2", "9", "x"])
    def test_bad_sizes_are_rejected_by_the_argument_parser(self, size: str) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--size", size], Session([]).read, Session([]).write)
        assert excinfo.value.code == 2

    def test_ctrl_c_exits_gracefully(self) -> None:
        def interrupt(_prompt: str) -> str:
            raise KeyboardInterrupt

        output: list[str] = []
        assert main(["-s", "3"], interrupt, output.append) == 130
        assert "Game abandoned." in "\n".join(output)


def test_running_the_module_as_a_script(tmp_path: object) -> None:
    """End to end through a real process, the way a person would run it."""
    completed = subprocess.run(
        [sys.executable, "-m", "tak.cli", "--size", "3"],
        input="\n".join(ROAD_GAME) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "White wins with a road" in completed.stdout
    assert "Result: R-0" in completed.stdout
    assert "Game saved to" in completed.stdout


def test_reader_type_matches_input() -> None:
    reader: Callable[[str], str] = input
    assert callable(reader)


class TestSaving:
    def saved(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.ptn"))

    def test_a_finished_game_is_saved_as_ptn(self, saves_dir: Path) -> None:
        session = Session(ROAD_GAME)
        assert main(["--size", "3"], session.read, session.write) == 0
        [path] = self.saved(saves_dir)
        assert path.name.endswith("_3x3_white-road.ptn")
        assert f"Game saved to {path}" in session.output
        loaded = game_from_ptn(path.read_text(encoding="utf-8"))
        assert loaded.result == GameResult(Color.WHITE, Reason.ROAD)
        assert loaded.ply == len(ROAD_GAME)

    def test_each_game_gets_its_own_file(self, saves_dir: Path) -> None:
        for _ in range(3):
            session = Session(ROAD_GAME)
            main(["--size", "3"], session.read, session.write)
        assert len(self.saved(saves_dir)) == 3

    def test_saves_dir_option(self, saves_dir: Path, tmp_path: Path) -> None:
        folder = tmp_path / "my games"
        session = Session(ROAD_GAME)
        main(["--size", "3", "--saves-dir", str(folder)], session.read, session.write)
        assert len(self.saved(folder)) == 1
        assert self.saved(saves_dir) == []

    def test_no_save_option(self, saves_dir: Path) -> None:
        session = Session(ROAD_GAME)
        main(["--size", "3", "--no-save"], session.read, session.write)
        assert self.saved(saves_dir) == []
        assert "saved" not in session.text

    def test_unfinished_games_are_not_saved(self, saves_dir: Path) -> None:
        for lines in (["a1", "quit"], ["a1"]):  # quitting, and input running out
            session = Session(lines)
            main(["--size", "3"], session.read, session.write)
        assert self.saved(saves_dir) == []

    def test_a_save_that_fails_does_not_crash_the_game(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("", encoding="utf-8")
        session = Session(ROAD_GAME)
        assert main(["--size", "3", "--saves-dir", str(blocker)], session.read, session.write) == 0
        assert "Result: R-0" in session.text
        assert "Could not save the game" in session.text

    def test_run_alone_does_not_save(self, saves_dir: Path) -> None:
        play_script(ROAD_GAME)
        assert self.saved(saves_dir) == []


class Scripted(Agent):
    """A computer player that plays the PTN moves it is given, in order."""

    key = "scripted"
    name = "Scripty"
    description = "plays the moves it is given"

    def __init__(self, moves: list[str]) -> None:
        super().__init__()
        self.moves = iter(moves)

    def pick(self, game: Game, rng: random.Random) -> Move:
        return parse_move(next(self.moves))


def play_computer(
    lines: list[str], computer: dict[Color, list[str]], save_dir: Path | None = None
) -> tuple[Game, GameResult | None, Session]:
    """Play on 3x3 against scripted computer players for the sides in ``computer``."""
    game = Game(3)
    session = Session(lines)
    players = {color: Scripted(moves) for color, moves in computer.items()}
    result = run(game, session.read, session.write, save_dir=save_dir, players=players)
    return game, result, session


class TestAgainstTheComputer:
    # On 3x3 White (the person) wins with a2-b2-c2; Black (the computer) plays a1, a3.
    HUMAN_WINS = (["a1", "b2", "c2"], {Color.BLACK: ["a2", "a3"]})
    # Black (the computer) wins with a2-b2-c2 (a2 placed for it by White) while the person
    # plays White.
    COMPUTER_WINS = (["a2", "a1", "b1"], {Color.BLACK: ["c3", "b2", "c2"]})

    def test_the_computer_answers_by_itself(self) -> None:
        game, result, session = play_computer(*self.HUMAN_WINS)
        assert result == GameResult(Color.WHITE, Reason.ROAD)
        assert len(session.prompts) == 3  # only the person's moves were asked for
        assert [played.move for played in game.history][1::2] == [
            parse_move("a2"),
            parse_move("a3"),
        ]

    def test_the_computer_is_introduced_and_its_moves_announced(self) -> None:
        _, _, session = play_computer(*self.HUMAN_WINS)
        assert "Black is played by the computer: Scripty, which plays the moves it is given." in (
            session.output
        )
        assert "Black (Scripty) places a white flat stone for you on a2." in session.output
        assert "Black (Scripty) plays a3." in session.output

    def test_the_board_is_shown_once_per_turn_not_before_the_computer_moves(self) -> None:
        _, _, session = play_computer(*self.HUMAN_WINS)
        assert session.text.count("Turn ") == 3  # once before each of the person's moves

    def test_who_won(self) -> None:
        _, _, session = play_computer(*self.HUMAN_WINS)
        assert session.output[-1] == "You win!"
        _, result, session = play_computer(*self.COMPUTER_WINS)
        assert result == GameResult(Color.BLACK, Reason.ROAD)
        assert session.output[-1] == "Scripty wins."

    def test_the_computer_can_move_first(self) -> None:
        game, _, session = play_computer(["c3", "quit"], {Color.WHITE: ["a1", "b1"]})
        assert game.ply == 3
        assert session.prompts == [
            "Black, place a white flat stone for your opponent (e.g. a1) > ",
            "Black > ",
        ]
        assert "White (Scripty) places a black flat stone for you on a1." in session.output

    def test_a_threat_made_by_the_person_is_called_before_the_reply(self) -> None:
        _, _, session = play_computer(["a1", "b2", "quit"], {Color.BLACK: ["a2", "c3"]})
        call = session.output.index("Tak!")  # b2 lets White win at c2
        assert session.output[call + 1] == "Black (Scripty) plays c3."

    def test_a_threat_made_by_the_computer_is_called(self) -> None:
        _, _, session = play_computer(["a2", "a1", "quit"], {Color.BLACK: ["c3", "b2"]})
        assert "Tak!" in session.text.split("Black (Scripty) plays b2.")[1]  # c2 wins next

    def test_undo_takes_back_the_reply_too(self) -> None:
        game, _, session = play_computer(
            ["a1", "b2", "undo", "c3", "quit"], {Color.BLACK: ["a2", "a3", "b3"]}
        )
        assert "Took back b2 and a3." in session.output
        assert [played.move for played in game.history] == [
            parse_move(move) for move in ("a1", "a2", "c3", "b3")
        ]

    @pytest.mark.parametrize("key", ["random", "greedy", "strategist"])
    def test_undo_and_replay_brings_the_same_reply(self, key: str) -> None:
        game = Game(5)
        placed: list[str] = []
        first_time: list[Move] = []
        steps = iter(["place", "place", "place", "undo", "undo", "1", "2", "quit"])

        def read(_prompt: str) -> str:
            step = next(steps)
            if step == "place":  # the first empty square, whatever the computer did
                board = game.board
                placed.append(next(s.name for s in board.squares() if not board.stack(s)))
                return placed[-1]
            if step == "undo" and not first_time:
                first_time.extend(played.move for played in game.history)
            if step.isdigit():  # play one of the moves taken back again
                return placed[int(step)]
            return step

        run(game, read, lambda _text: None, players={Color.BLACK: make_agent(key)})
        assert len(first_time) == 6
        assert [played.move for played in game.history] == first_time

    def test_undo_when_only_the_computer_has_moved(self) -> None:
        game, _, session = play_computer(["undo", "quit"], {Color.WHITE: ["a1"]})
        assert "There is nothing to undo." in session.output
        assert game.ply == 1

    def test_undo_back_to_the_start(self) -> None:
        game, _, session = play_computer(["c3", "undo", "quit"], {Color.WHITE: ["a1", "b1"]})
        assert "Took back c3 and b1." in session.output
        assert game.ply == 1  # the computer's opening move stays; it's the person's turn

    def test_help_explains_playing_the_computer(self) -> None:
        _, _, session = play_computer(["help", "quit"], {Color.BLACK: []})
        assert "undo takes\nback your last move and the computer's reply" in session.text

    def test_the_computer_can_play_both_sides(self) -> None:
        white = ["a1", "b2", "c2"]
        game, result, session = play_computer([], {Color.WHITE: white, Color.BLACK: ["a2", "a3"]})
        assert result == GameResult(Color.WHITE, Reason.ROAD)
        assert session.prompts == []
        assert "White (Scripty) places a black flat stone for Scripty on a1." in session.output
        assert "You win!" not in session.text and "Scripty wins." not in session.text

    def test_saved_games_name_the_players(self, tmp_path: Path) -> None:
        play_computer(*self.HUMAN_WINS, save_dir=tmp_path)
        text = next(tmp_path.glob("*.ptn")).read_text(encoding="utf-8")
        assert '[Player1 "Human"]' in text
        assert '[Player2 "Scripty (computer)"]' in text
        assert game_from_ptn(text).result == GameResult(Color.WHITE, Reason.ROAD)
