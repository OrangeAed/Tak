"""Announcing threats in the terminal game: "Tak!" and "Tinue!".

After a move that threatens to win next turn, the status printed before the
next prompt carries a "Tak!" line, or "Tinue!" if the threat can't be
stopped, placed just before the "Turn N: Player to move." line.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import pytest

from tak.cli.app import threat_call
from tak.cli.render import render_status
from tak.engine import Color, Game, GameResult, Reason

Started = Callable[..., Game]
PlayScript = Callable[..., tuple[GameResult | None, list[str]]]

CALLS = {"Tak!", "Tinue!"}
# 3x3 scripts, moves for both players in order.
TAK_GAME = ["a1", "c3", "b1", "b3", "b2"]  # White's b2 threatens c2 (b1-b2-c2-c3)
TINUE_GAME = ["b3", "c1", "c2", "b1", "b2"]  # White's b2 threatens both c3 and a2
BLACK_TINUE_GAME = ["b3", "a2", "a3", "a1", "c3", "b1"]  # Black's b1 threatens c1 and b2


def calls_made(lines: list[str]) -> list[tuple[str, str]]:
    """Each call printed, paired with the line that follows it."""
    return [(line, lines[i + 1]) for i, line in enumerate(lines) if line in CALLS]


class TestRenderStatus:
    def test_a_call_goes_just_before_the_turn_line(self) -> None:
        for call in ["Tak!", "Tinue!"]:
            lines = render_status(Game(3), call).splitlines()
            assert lines[1:] == [call, "Turn 1: White to move."]

    def test_no_call_no_extra_line(self) -> None:
        assert len(render_status(Game(3)).splitlines()) == 2
        assert render_status(Game(3), None) == render_status(Game(3))


class TestThreatCall:
    def test_nothing_to_announce(self, started: Started) -> None:
        assert threat_call(started("a1 c3 b1 b3", size=3), Color.BLACK) is None

    def test_tak(self, started: Started) -> None:
        assert threat_call(started(" ".join(TAK_GAME), size=3), Color.WHITE) == "Tak!"

    def test_tinue(self, started: Started) -> None:
        assert threat_call(started(" ".join(TINUE_GAME), size=3), Color.WHITE) == "Tinue!"

    def test_tinue_for_black(self, started: Started) -> None:
        game = started(" ".join(BLACK_TINUE_GAME), size=3)
        assert threat_call(game, Color.BLACK) == "Tinue!"


class TestGameLoop:
    def test_tak_is_printed_right_before_the_next_turn_line(self, play_script: PlayScript) -> None:
        _, lines = play_script([*TAK_GAME, "quit"])
        assert calls_made(lines) == [("Tak!", "Turn 3: Black to move.")]
        tak = lines.index("Tak!")
        assert lines[tak - 1].startswith("White: 7 stones")  # the reserves line comes first

    def test_no_call_when_nothing_is_threatened(self, play_script: PlayScript) -> None:
        _, lines = play_script([*TAK_GAME[:4], "quit"])
        assert calls_made(lines) == []

    def test_an_unstoppable_threat_prints_tinue_instead_of_tak(
        self, play_script: PlayScript
    ) -> None:
        _, lines = play_script([*TINUE_GAME, "quit"])
        assert calls_made(lines) == [
            ("Tak!", "Turn 2: Black to move."),  # White's c2 threatened c3
            ("Tak!", "Turn 3: White to move."),  # Black's b1 threatened b2
            ("Tinue!", "Turn 3: Black to move."),
        ]

    def test_tinue_for_black(self, play_script: PlayScript) -> None:
        _, lines = play_script([*BLACK_TINUE_GAME, "quit"])
        assert calls_made(lines)[-1] == ("Tinue!", "Turn 4: White to move.")

    def test_tak_on_a_bigger_board(self, play_script: PlayScript) -> None:
        # 5x5: Black lines up a5-b5-c5-d5, threatening e5; White's stones are scattered.
        script = ["a5", "e1", "c3", "b5", "e3", "c5", "a1", "d5", "quit"]
        _, lines = play_script(script, size=5)
        assert calls_made(lines) == [("Tak!", "Turn 5: White to move.")]

    @pytest.mark.parametrize(
        ("script", "finish", "call"),
        [(TAK_GAME, ["c1", "c2"], "Tak!"), (TINUE_GAME, ["c3", "a2"], "Tinue!")],
    )
    def test_the_threat_carried_out(
        self, play_script: PlayScript, script: list[str], finish: list[str], call: str
    ) -> None:
        result, lines = play_script([*script, *finish])
        assert result == GameResult(Color.WHITE, Reason.ROAD)
        assert lines.count(call) == 1
        # Nothing is announced after the winning move; the game is simply over.
        game_over = next(i for i, line in enumerate(lines) if line.startswith("Game over"))
        last_turn = max(i for i, line in enumerate(lines) if line.startswith("Turn "))
        assert not any(line in CALLS for line in lines[last_turn + 1 : game_over])

    @pytest.mark.parametrize(("script", "call"), [(TAK_GAME, "Tak!"), (TINUE_GAME, "Tinue!")])
    def test_a_call_is_made_once_for_the_move(
        self, play_script: PlayScript, script: list[str], call: str
    ) -> None:
        # A rejected move, a command, or a blank line doesn't repeat the call.
        _, lines = play_script([*script, "b2", "help", "moves", "", "quit"])
        assert "Illegal move: b2 is already occupied." in lines
        assert lines.count(call) == 1

    def test_undo_makes_no_call_but_replaying_the_move_does(self, play_script: PlayScript) -> None:
        _, lines = play_script([*TAK_GAME, "undo", "b2", "quit"])
        assert "Took back b2." in lines
        assert calls_made(lines) == [
            ("Tak!", "Turn 3: Black to move."),  # b2 played
            ("Tak!", "Turn 3: Black to move."),  # b2 played again after the undo
        ]
        undo = lines.index("Took back b2.")
        after_undo = lines[undo : lines.index("Turn 3: White to move.", undo) + 1]
        assert not any(line in CALLS for line in after_undo)


def test_calls_in_a_real_terminal_session() -> None:
    """End to end through a real process, the way a person would run it."""
    completed = subprocess.run(
        [sys.executable, "-m", "tak.cli", "--size", "3"],
        input="\n".join([*TINUE_GAME, "c3", "a2"]) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert calls_made(completed.stdout.splitlines()) == [
        ("Tak!", "Turn 2: Black to move."),
        ("Tak!", "Turn 3: White to move."),
        ("Tinue!", "Turn 3: Black to move."),
        ("Tak!", "Turn 4: White to move."),  # Black's c3 blocks, and threatens a3
    ]
    assert "White wins with a road: a2 -> b2 -> c2" in completed.stdout
