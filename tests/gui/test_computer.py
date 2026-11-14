"""Working out the computer's move in the background (no drawing involved)."""

from __future__ import annotations

import threading
import time

import pytest

from tak.ai import Agent
from tak.engine import Game, Move, parse_move
from tak.gui.computer import ComputerTurn, position_of

from .conftest import Scripted


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def wait_for(turn: ComputerTurn, timeout: float = 5.0) -> Move:
    """Poll like the window does, until the move is ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        move = turn.result()
        if move is not None:
            return move
        time.sleep(0.005)
    raise AssertionError("the computer never moved")


def test_position_of_lists_the_moves_played() -> None:
    game = Game(3)
    assert position_of(game) == ()
    game.play(parse_move("a1"))
    assert position_of(game) == (parse_move("a1"),)


class TestWithoutAThread:
    def test_the_move_is_ready_at_once(self) -> None:
        turn = ComputerTurn(min_thinking=0, threaded=False)
        assert not turn.busy
        turn.start(Scripted("a1"), Game(3))
        assert turn.busy and turn.position == ()
        assert turn.result() == parse_move("a1")
        assert not turn.busy
        assert turn.result() is None  # collected once only

    def test_the_move_waits_for_the_minimum_thinking_time(self) -> None:
        clock = FakeClock()
        turn = ComputerTurn(min_thinking=0.5, threaded=False, clock=clock)
        turn.start(Scripted("a1"), Game(3))
        assert turn.result() is None
        clock.now = 0.49
        assert turn.result() is None
        clock.now = 0.5
        assert turn.result() == parse_move("a1")

    def test_errors_reach_the_caller(self) -> None:
        turn = ComputerTurn(min_thinking=0, threaded=False)
        turn.start(Scripted(), Game(3))  # no moves to give
        with pytest.raises(RuntimeError, match="out of moves"):
            turn.result()
        assert not turn.busy

    def test_the_agent_thinks_on_a_copy(self) -> None:
        class Meddler(Agent):
            key = name = description = "meddler"

            def pick(self, game: Game) -> Move:
                game.play(parse_move("c3"))  # scribbles on the game it was given
                return parse_move("b2")

        game = Game(3)
        turn = ComputerTurn(min_thinking=0, threaded=False)
        turn.start(Meddler(), game)
        assert turn.result() == parse_move("b2")
        assert game.ply == 0


class TestInAThread:
    def test_thinking_happens_in_the_background(self) -> None:
        gate = threading.Event()
        turn = ComputerTurn(min_thinking=0)
        turn.start(Scripted("a1", gate=gate), Game(3))
        time.sleep(0.02)
        assert turn.busy and turn.result() is None  # still thinking
        gate.set()
        assert wait_for(turn) == parse_move("a1")
        assert not turn.busy

    def test_a_cancelled_move_is_thrown_away(self) -> None:
        gate = threading.Event()
        turn = ComputerTurn(min_thinking=0)
        turn.start(Scripted("a1", gate=gate), Game(3))
        turn.cancel()
        assert not turn.busy
        gate.set()
        time.sleep(0.05)
        assert turn.result() is None

    def test_starting_again_drops_the_earlier_move(self) -> None:
        gate = threading.Event()
        turn = ComputerTurn(min_thinking=0)
        turn.start(Scripted("a1", gate=gate), Game(3))
        later = Game(3)
        later.play(parse_move("c3"))
        turn.start(Scripted("b2"), later)
        assert turn.position == (parse_move("c3"),)
        gate.set()  # the first agent finishes last, but its answer is stale
        assert wait_for(turn) == parse_move("b2")
        time.sleep(0.05)
        assert turn.result() is None
