"""Letting a computer player think without freezing the window.

Choosing a move can take the stronger players a second or so on big boards.
The window has to keep drawing and answering clicks meanwhile, so
:class:`ComputerTurn` runs the agent in a background thread on its own copy
of the game, and the window polls it once a frame. No drawing happens here.

A move is only handed over once a short minimum time has passed, so that the
person's own move visibly lands before the reply appears. And a move is only
any use for the position it was worked out for: if the game has moved on
(say, the person stepped back), the window cancels it and the answer is thrown
away.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from tak.ai.base import Agent
from tak.engine.game import Game
from tak.engine.moves import Move

#: The least time between the computer starting to think and its move appearing.
MIN_THINKING = 0.5

Position = tuple[Move, ...]


def position_of(game: Game) -> Position:
    """Identifies a game's position: the moves played to reach it."""
    return tuple(played.move for played in game.history)


class ComputerTurn:
    """One computer move being worked out, at most, at a time.

    With ``threaded`` off, the move is worked out straight away inside
    :meth:`start`, which is simpler for tests.
    """

    def __init__(
        self,
        min_thinking: float = MIN_THINKING,
        threaded: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_thinking = min_thinking
        self.threaded = threaded
        self.clock = clock
        self.agent: Agent | None = None
        self.position: Position | None = None  # what the move being worked out is for
        self._started = 0.0
        self._lock = threading.Lock()
        self._move: Move | None = None
        self._error: BaseException | None = None
        self._job = 0  # bumped on every start and cancel, so stale threads are ignored

    @property
    def busy(self) -> bool:
        """Whether a move is being worked out (or is ready but not yet collected)."""
        return self.position is not None

    def start(self, agent: Agent, game: Game) -> None:
        """Start working out ``agent``'s move in ``game`` (any earlier one is dropped)."""
        with self._lock:
            self._job += 1
            job = self._job
            self._move = None
            self._error = None
        self.agent = agent
        self.position = position_of(game)
        self._started = self.clock()
        copy = game.copy()
        if self.threaded:
            worker = threading.Thread(
                target=self._think, args=(job, agent, copy), name="tak-computer", daemon=True
            )
            worker.start()
        else:
            self._think(job, agent, copy)

    def _think(self, job: int, agent: Agent, game: Game) -> None:
        move: Move | None = None
        error: BaseException | None = None
        try:
            move = agent.choose_move(game)
        except Exception as exc:  # reported to the window rather than lost in the thread
            error = exc
        with self._lock:
            if job == self._job:
                self._move, self._error = move, error

    def cancel(self) -> None:
        """Forget the move being worked out; the thread's answer will be ignored."""
        with self._lock:
            self._job += 1
            self._move = None
            self._error = None
        self.agent = None
        self.position = None

    def result(self) -> Move | None:
        """The move, once it's ready and the minimum time has passed; ``None`` until then.

        Collecting the move ends the turn (:attr:`busy` goes back to False).
        Re-raises, in the caller's thread, anything the agent raised.
        """
        if self.position is None or self.clock() - self._started < self.min_thinking:
            return None
        with self._lock:
            move, error = self._move, self._error
        if error is not None:
            self.cancel()
            raise error
        if move is not None:
            self.agent = None
            self.position = None
        return move
