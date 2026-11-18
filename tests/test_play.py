"""Playing against the computer: ``python -m tak.play`` and ``play_against_bot``."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

from tak import play
from tak.ai import GreedyAgent, RandomAgent, StrategistAgent
from tak.engine import Color, game_from_ptn
from tak.play import choose_color, main, play_against_bot


class FirstMovePlayer:
    """Stands in for the person: asks for the legal moves, then plays the first one."""

    def __init__(self, limit: int = 1000) -> None:
        self.output: list[str] = []
        self.prompts: list[str] = []
        self.limit = limit
        self._asked = False  # "moves" was just sent; the next line written lists them
        self._first_move: str | None = None

    def write(self, text: str) -> None:
        if self._asked and self._first_move is None:
            self._first_move = text.split()[0]
        self.output.append(text)

    def read(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) > self.limit:
            return "quit"
        if self._first_move is None:
            self._asked = True
            return "moves"
        move, self._first_move, self._asked = self._first_move, None, False
        return move

    @property
    def text(self) -> str:
        return "\n".join(self.output)


class TestChooseColor:
    def test_named_colors(self) -> None:
        assert choose_color("white") is Color.WHITE
        assert choose_color("Black") is Color.BLACK

    def test_random_picks_both(self) -> None:
        rng = random.Random(0)
        assert {choose_color("random", rng) for _ in range(20)} == set(Color)

    def test_anything_else_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="white, black, random"):
            choose_color("green")


class TestPlayAgainstBot:
    @pytest.mark.parametrize("bot", ["random", "greedy", "strategist"])
    @pytest.mark.parametrize("color", ["white", "black"])
    def test_a_whole_game_in_the_terminal(self, bot: str, color: str) -> None:
        person = FirstMovePlayer()
        result = play_against_bot(
            bot, color, 4, seed=1, save=False, read=person.read, write=person.write
        )
        assert result is not None
        name = {"random": "Random", "greedy": "Greedy", "strategist": "Strategist"}[bot]
        assert person.output[0] == f"You play {color.capitalize()} against {name}."
        computer = "Black" if color == "white" else "White"
        assert f"{computer} ({name}) plays" in person.text
        assert "Game over" in person.text

    def test_you_move_first_as_white_and_second_as_black(self) -> None:
        white = FirstMovePlayer(limit=0)
        play_against_bot("random", "white", 3, save=False, read=white.read, write=white.write)
        assert white.prompts[0].startswith("White, place a black flat")
        black = FirstMovePlayer(limit=0)
        play_against_bot("random", "black", 3, save=False, read=black.read, write=black.write)
        assert black.prompts[0].startswith("Black, place a white flat")
        assert "White (Random) places a black flat stone for you" in black.text

    def test_a_random_color_says_so(self) -> None:
        person = FirstMovePlayer(limit=0)
        play_against_bot("greedy", "random", 3, seed=4, read=person.read, write=person.write)
        assert "(picked at random)" in person.output[0]

    def test_the_board_size(self) -> None:
        person = FirstMovePlayer(limit=0)
        play_against_bot("greedy", "white", 7, save=False, read=person.read, write=person.write)
        assert "Tak on a 7x7 board." in person.text

    def test_the_same_seed_plays_the_same_game(self) -> None:
        def transcript() -> str:
            person = FirstMovePlayer()
            play_against_bot(
                "greedy", "random", 4, seed=9, save=False, read=person.read, write=person.write
            )
            return person.text

        assert transcript() == transcript()

    def test_the_finished_game_is_saved_with_the_players(self, tmp_path: Path) -> None:
        person = FirstMovePlayer()
        play_against_bot(
            "strategist",
            "black",
            3,
            seed=2,
            save_dir=tmp_path,
            read=person.read,
            write=person.write,
        )
        [saved] = tmp_path.glob("*.ptn")
        text = saved.read_text(encoding="utf-8")
        assert '[Player1 "Strategist (computer)"]' in text
        assert '[Player2 "Human"]' in text
        assert game_from_ptn(text).is_over

    def test_not_saving(self, saves_dir: Path) -> None:
        person = FirstMovePlayer()
        play_against_bot(
            "random", "white", 3, seed=2, save=False, read=person.read, write=person.write
        )
        assert list(saves_dir.glob("*.ptn")) == []

    @pytest.mark.parametrize(
        ("options", "problem"),
        [
            ({"bot": "minimax"}, "unknown computer player"),
            ({"color": "green"}, "unknown color"),
            ({"size": 9}, "unsupported board size"),
        ],
    )
    def test_bad_options(self, options: dict[str, object], problem: str) -> None:
        with pytest.raises(ValueError, match=problem):
            play_against_bot(**options)  # type: ignore[arg-type]

    def test_the_window(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        opened = []
        monkeypatch.setattr(
            "tak.gui.app.run",
            lambda session, message, eval_bar: opened.append((session, message, eval_bar)),
        )
        assert play_against_bot("strategist", "black", 6, gui=True, save_dir=tmp_path) is None
        [(session, message, eval_bar)] = opened
        assert eval_bar is False
        assert session.game.size == 6
        assert list(session.players) == [Color.WHITE]
        assert isinstance(session.players[Color.WHITE], StrategistAgent)
        assert session.recorder.save_dir == tmp_path
        assert message == ("info", "You play Black against Strategist.")

    def test_the_window_with_the_evaluation_bar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened = []
        monkeypatch.setattr(
            "tak.gui.app.run", lambda session, message, eval_bar: opened.append(eval_bar)
        )
        play_against_bot("learner", gui=True, eval_bar=True, save=False)
        assert opened == [True]

    def test_the_evaluation_bar_in_the_terminal(self) -> None:
        person = FirstMovePlayer(limit=6)
        play_against_bot(
            "learner",
            "white",
            4,
            eval_bar=True,
            seed=2,
            save=False,
            read=person.read,
            write=person.write,
        )
        assert " |w|" in person.text and " |b|" in person.text
        assert "Evaluation: " in person.text


class TestMain:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr(play, "play_against_bot", lambda *a, **k: calls.append((a, k)))
        assert main([]) == 0
        [(args, options)] = calls
        assert args == ("greedy", "white", 5)
        assert options["gui"] is False
        assert options["eval_bar"] is False

    def test_the_evaluation_bar_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr(play, "play_against_bot", lambda *a, **k: calls.append((a, k)))
        main(["--eval-bar", "--gui"])
        [(_, options)] = calls
        assert options["eval_bar"] is True and options["gui"] is True

    def test_options(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls = []
        monkeypatch.setattr(play, "play_against_bot", lambda *a, **k: calls.append((a, k)))
        argv = ["--bot", "strategist", "--color", "RANDOM", "--size", "6", "--seed", "3"]
        assert main([*argv, "--saves-dir", str(tmp_path)]) == 0
        [(args, options)] = calls
        assert args == ("strategist", "random", 6)
        assert options["seed"] == 3 and options["save_dir"] == tmp_path

    def test_no_save(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr(play, "play_against_bot", lambda *a, **k: calls.append((a, k)))
        main(["-b", "random", "-c", "black", "-s", "3", "--no-save"])
        [(args, options)] = calls
        assert args == ("random", "black", 3)
        assert options["save"] is False

    @pytest.mark.parametrize("argv", [["--bot", "minimax"], ["--color", "green"], ["--size", "9"]])
    def test_bad_options_are_rejected(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2

    def test_the_help_lists_the_computer_players(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        help_text = capsys.readouterr().out
        for kind in (RandomAgent, GreedyAgent, StrategistAgent):
            assert kind.key in help_text and kind.description in help_text

    def test_plays_in_the_terminal(self) -> None:
        person = FirstMovePlayer()
        assert main(["--size", "3", "--seed", "5", "--no-save"], person.read, person.write) == 0
        assert "Game over" in person.text

    def test_the_window_needs_pygame(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setitem(sys.modules, "pygame", None)  # as if it weren't installed
        assert main(["--gui"]) == 1
        assert "needs pygame" in capsys.readouterr().err

    def test_ctrl_c_exits_gracefully(self) -> None:
        def interrupt(_prompt: str) -> str:
            raise KeyboardInterrupt

        output: list[str] = []
        assert main(["-s", "3"], interrupt, output.append) == 130
        assert "Game abandoned." in output


def test_running_the_module_as_a_script() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tak.play",
            "--bot",
            "strategist",
            "--color",
            "black",
            "--seed",
            "1",
        ],
        input="quit\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "You play Black against Strategist." in completed.stdout
    assert "White (Strategist) places a black flat stone for you on" in completed.stdout
    assert "Game abandoned." in completed.stdout
