"""Tests for announcing threats in the terminal game.

After a move that threatens to win next turn, "Tak!" is printed just before
the "Turn {x}: {Player} to move." line; if the threat can't be stopped,
"Tinue!" is printed instead.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

from tak.cli.app import threat_call
from tak.cli.render import render_status
from tak.engine import Color, Game, GameResult, Reason

Started = Callable[..., Game]
PlayScript = Callable[..., tuple[Game, GameResult | None, list[str]]]

CALLS = {"Tak!", "Tinue!"}
TAK_GAME = ["a1", "c3", "b1", "b3", "b2"]  # 3x3: b2 threatens c2, joining b1-b2-c2-c3
# 3x3: after Tak from each side, White's b2 blocks Black and threatens both c3 and a2.
TINUE_GAME = ["b3", "c1", "c2", "b1", "b2"]


def calls_in(lines: list[str]) -> list[tuple[str, str]]:
    """Each announcement, paired with the line printed after it."""
    return [(line, lines[i + 1]) for i, line in enumerate(lines) if line in CALLS]


class TestStatus:
    def test_a_call_goes_just_before_the_turn_line(self) -> None:
        for call in sorted(CALLS):
            lines = render_status(Game(3), call).splitlines()
            assert lines[1:] == [call, "Turn 1: White to move."]

    def test_no_call_no_extra_line(self) -> None:
        assert len(render_status(Game(3)).splitlines()) == 2


class TestThreatCall:
    def test_nothing_to_announce(self, started: Started) -> None:
        assert threat_call(started("a1 c3 b1 b3", size=3), Color.BLACK) is None

    def test_tak(self, started: Started) -> None:
        assert threat_call(started(" ".join(TAK_GAME), size=3), Color.WHITE) == "Tak!"

    def test_tinue(self, started: Started) -> None:
        assert threat_call(started(" ".join(TINUE_GAME), size=3), Color.WHITE) == "Tinue!"

    def test_nothing_once_the_game_is_over(self, started: Started) -> None:
        assert threat_call(started("a1 a2 b2 a3 c2", size=3), Color.WHITE) is None


class TestTak:
    def test_printed_right_before_the_next_turn_line(self, play_script: PlayScript) -> None:
        _, _, lines = play_script([*TAK_GAME, "quit"])
        assert calls_in(lines) == [("Tak!", "Turn 3: Black to move.")]
        tak = lines.index("Tak!")
        assert lines[tak - 1].startswith("White: 7 stones")  # the reserves line comes first

    def test_announced_for_black_too(self, play_script: PlayScript) -> None:
        # 5x5: Black lines up a5-b5-c5-d5, threatening e5; White's stones are scattered.
        script = ["a5", "e1", "c3", "b5", "e3", "c5", "a1", "d5", "quit"]
        _, _, lines = play_script(script, size=5)
        assert calls_in(lines) == [("Tak!", "Turn 5: White to move.")]

    def test_no_call_when_nothing_is_threatened(self, play_script: PlayScript) -> None:
        _, _, lines = play_script([*TAK_GAME[:4], "quit"])
        assert calls_in(lines) == []

    def test_followed_by_the_win(self, play_script: PlayScript) -> None:
        # Black doesn't block c2. The winning move itself brings no call.
        _, result, lines = play_script([*TAK_GAME, "c1", "c2"])
        assert result == GameResult(Color.WHITE, Reason.ROAD)
        assert [call for call, _ in calls_in(lines)] == ["Tak!"]

    def test_announced_once_for_the_move_that_made_it(self, play_script: PlayScript) -> None:
        # A rejected move, a command, or a blank line doesn't repeat the announcement.
        _, _, lines = play_script([*TAK_GAME, "b2", "help", "", "quit"])
        assert "Illegal move: b2 is already occupied." in lines
        assert lines.count("Tak!") == 1

    def test_undo_takes_the_call_back_and_replaying_repeats_it(
        self, play_script: PlayScript
    ) -> None:
        _, _, lines = play_script([*TAK_GAME, "undo", "b2", "quit"])
        assert "Took back b2." in lines
        assert calls_in(lines) == [
            ("Tak!", "Turn 3: Black to move."),
            ("Tak!", "Turn 3: Black to move."),  # after b2 is played again
        ]
        # The status shown straight after the undo (White to move again) has no call.
        undone = lines.index("Took back b2.")
        turn_after_undo = lines.index("Turn 3: White to move.", undone)
        assert lines[turn_after_undo - 1] not in CALLS


class TestTinue:
    def test_printed_instead_of_tak(self, play_script: PlayScript) -> None:
        _, _, lines = play_script([*TINUE_GAME, "quit"])
        assert calls_in(lines) == [
            ("Tak!", "Turn 2: Black to move."),  # White's c2 threatened c3
            ("Tak!", "Turn 3: White to move."),  # Black's b1 threatened b2
            ("Tinue!", "Turn 3: Black to move."),
        ]

    def test_followed_by_the_win(self, play_script: PlayScript) -> None:
        # Black blocks c3 (which threatens a3), and White wins with a2.
        _, result, lines = play_script([*TINUE_GAME, "c3", "a2"])
        assert result == GameResult(Color.WHITE, Reason.ROAD)
        assert lines.count("Tinue!") == 1


class TestBlocksThatOpenNewThreats:
    """A reply that stops every threat but opens a new one."""

    # 3x3: White's a1 threatens a2. Black's b2< would block a2 but uncover
    # White's flat on b2, yet Black has safe blocks (a2, Sa2, 2b2<): Tak only.
    TRAP_GAME = ["c2", "b2", "a3", "c2<", "a1"]

    def test_a_trap_is_only_tak(self, play_script: PlayScript) -> None:
        _, _, lines = play_script([*self.TRAP_GAME, "quit"])
        assert calls_in(lines) == [("Tak!", "Turn 3: Black to move.")]

    def test_falling_into_the_trap_loses(self, play_script: PlayScript) -> None:
        _, result, lines = play_script([*self.TRAP_GAME, "b2<", "b2<"])
        assert result == GameResult(Color.WHITE, Reason.ROAD)
        assert [call for call, _ in calls_in(lines)] == ["Tak!"]

    # 3x3: Black's c2 threatens row 2 and, by capturing c3, row 3. White's only
    # way to stop both (c3-, capturing c2) leaves c3 empty for Black: Tinue.
    TINUE_GAME = ["a3", "c2", "a1", "b3", "Sb1", "a2", "c2+", "c2"]

    def test_when_the_only_block_backfires_it_is_tinue(self, play_script: PlayScript) -> None:
        _, result, lines = play_script([*self.TINUE_GAME, "c3-", "c3"])
        assert calls_in(lines) == [
            ("Tak!", "Turn 3: White to move."),  # Black's b3 threatened c3
            ("Tak!", "Turn 4: White to move."),  # Black's a2, and c3 is still open
            ("Tinue!", "Turn 5: White to move."),
        ]
        assert result == GameResult(Color.BLACK, Reason.ROAD)


def test_calls_in_the_real_program() -> None:
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
    lines = completed.stdout.splitlines()
    assert ("Tinue!", "Turn 3: Black to move.") in calls_in(lines)
    assert "White wins with a road" in completed.stdout
