"""A game in the window: playing moves, stepping through them, loading, and saving.

Pure logic with no drawing, so it can be tested on its own.

The session keeps the moves after the current position as well as before it,
so you can step back and forward through a game (one you are playing, or one
loaded from a file) without losing anything. Playing a new move part-way
through replaces the moves that came after it, like typing over an undo in an
editor. For a game loaded from a file that would throw away part of the game
being reviewed, so it asks for confirmation first; the file itself is never
changed.

A session can also have computer players (:mod:`tak.ai`) for one or both
sides. The session refuses moves made for them by hand, and plays the moves
they choose with :meth:`GameSession.play_computer`. Working out the move is
left to the window (see :mod:`tak.gui.computer`), which does it in the
background. The computer only moves at the latest position: while you step
back through the game it waits, and if you play a new move from an earlier
position, it answers that.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tak.ai.base import Agent, player_headers
from tak.engine.game import Game
from tak.engine.moves import Move
from tak.engine.notation import NotationError, format_move, game_from_ptn, parse_move
from tak.engine.pieces import Color
from tak.engine.rules import IllegalMoveError
from tak.saves import save_finished_game

#: Message kinds, used to pick a color when drawing.
INFO, GOOD, BAD = "info", "good", "bad"

Feedback = tuple[str, str]  # (kind, message)


def threat_call(game: Game, mover: Color) -> str | None:
    """What to announce after ``mover``'s move: "Tinue!", "Tak!", or nothing."""
    if not game.threatens_win(mover):
        return None
    return "Tinue!" if game.is_tinue(mover) else "Tak!"


class GameRecorder:
    """Saves the game each time it ends, keeping to one file per game.

    If the game is finished again after stepping back and playing differently,
    the file saved the first time is rewritten with the new ending instead of
    a second file being made.
    """

    def __init__(self, save_dir: Path | None, headers: Mapping[str, str] | None = None) -> None:
        self.save_dir = save_dir
        self.headers = dict(headers or {})  # extra PTN headers, such as who played
        self.path: Path | None = None

    def after_move(self, game: Game) -> Feedback | None:
        """Save if ``game`` just ended; return a ``(kind, message)`` to show, if any."""
        if not game.is_over or self.save_dir is None:
            return None
        path, message = save_finished_game(
            game, self.save_dir, replace=self.path, headers=self.headers
        )
        if path is None:
            return BAD, message
        self.path = path
        return INFO, message


class GameSession:
    """One game and the timeline of its moves, with the current position somewhere on it."""

    def __init__(
        self,
        game: Game,
        save_dir: Path | None = None,
        players: Mapping[Color, Agent] | None = None,
    ) -> None:
        self.game = game
        self.players = dict(players or {})  # the sides the computer plays
        self.recorder = GameRecorder(save_dir, player_headers(self.players))
        self.future: list[Move] = []  # moves after the current position, next one last
        self.source: Path | None = None  # the file this game was loaded from, if any
        self.pending: Move | None = None  # a move waiting for confirmation

    @classmethod
    def load(cls, path: Path, save_dir: Path | None = None) -> GameSession:
        """Open a PTN file, positioned at the start so it can be stepped through.

        Raises :class:`OSError`, :class:`NotationError`, or
        :class:`IllegalMoveError` if the file can't be read or replayed.
        """
        game = game_from_ptn(path.read_text(encoding="utf-8"))
        session = cls(game, save_dir)
        session.source = path
        session.to_start()
        return session

    # --- Where we are -------------------------------------------------------------

    @property
    def total(self) -> int:
        """How many moves the timeline holds, before and after the current position."""
        return self.game.ply + len(self.future)

    @property
    def last_move(self) -> Move | None:
        return self.game.history[-1].move if self.game.history else None

    def position_text(self) -> str:
        """For example ``Move 12 of 37: 3c3>12``."""
        text = f"Move {self.game.ply} of {self.total}"
        if self.game.history:
            played = self.game.history[-1]
            text += f": {format_move(played.move, flattened=played.flattened)}"
        return text

    @property
    def computer_to_move(self) -> Agent | None:
        """The computer player whose turn it is here, or ``None`` (a person's turn, or over)."""
        if self.game.is_over:
            return None
        return self.players.get(self.game.to_move)

    @property
    def waiting_for_computer(self) -> bool:
        """Whether the computer should move now: it's its turn at the latest position."""
        return self.computer_to_move is not None and not self.future

    def versus_text(self) -> str | None:
        """Who plays whom, e.g. ``You play Black against Strategist``; ``None`` if no computer."""
        if not self.players:
            return None
        people = [color for color in Color if color not in self.players]
        if not people:
            return " vs ".join(f"{self.players[c].name} ({c})" for c in Color)
        names = " and ".join(agent.name for agent in self.players.values())
        return f"You play {people[0]} against {names}"

    def computer_turn(self) -> Feedback | None:
        """If the computer is to move here, a message saying a person can't; else ``None``."""
        agent = self.computer_to_move
        if agent is None:
            return None
        color = self.game.to_move
        if self.future:
            return (
                BAD,
                f"{agent.name} plays {color}, so you can't move here. Step forward, or back "
                "to one of your turns to play on from there.",
            )
        return BAD, f"It's {agent.name}'s turn ({color}). Wait for it to move."

    # --- Stepping ----------------------------------------------------------------

    def back(self) -> bool:
        """Step back one move; returns False at the start."""
        if not self.game.history:
            return False
        self.pending = None
        self.future.append(self.game.undo())
        return True

    def forward(self) -> bool:
        """Step forward one move; returns False at the end."""
        if not self.future:
            return False
        self.pending = None
        self.game.play(self.future.pop())
        return True

    def to_start(self) -> None:
        while self.back():
            pass

    def to_end(self) -> None:
        while self.forward():
            pass

    # --- Playing ----------------------------------------------------------------

    def play(self, move: Move) -> Feedback | None:
        """Play a person's new move here; return a ``(kind, message)`` to show, if any."""
        refusal = self.computer_turn()
        if refusal is not None:
            return refusal
        if self.future and self.source is not None and self.pending != move:
            try:
                self.game.check(move)  # don't ask to confirm a move that can't be played
            except IllegalMoveError as exc:
                return BAD, f"Illegal move: {exc}."
            self.pending = move
            later = len(self.future)
            return (
                INFO,
                f"Playing {format_move(move)} here replaces the {later} later "
                f"move{'s' if later != 1 else ''} of {self.source.name} (the file "
                "isn't changed). Press Enter to play it, or Esc to cancel.",
            )
        self.pending = None
        mover = self.game.to_move
        try:
            self.game.play(move)
        except IllegalMoveError as exc:
            return BAD, f"Illegal move: {exc}."
        if self.future:
            self.future.clear()
            self.source = None  # it's a different game from the file now
        saved = self.recorder.after_move(self.game)
        if saved:
            return saved
        call = threat_call(self.game, mover)
        return (GOOD, call) if call else None

    def play_computer(self, move: Move) -> Feedback:
        """Play the move the computer chose, at the latest position.

        Raises :class:`ValueError` if it isn't the computer's turn there, and
        :class:`IllegalMoveError` if the move isn't legal.
        """
        agent = self.computer_to_move
        if agent is None or self.future:
            raise ValueError("it isn't the computer's turn at the latest position")
        mover = self.game.to_move
        opening = self.game.in_opening
        self.pending = None
        self.game.play(move)
        played = self.game.history[-1]
        move_text = format_move(played.move, flattened=played.flattened)
        if opening:
            other = self.players.get(mover.opponent)
            whom = other.name if other else "you"
            text = f"{agent.name} placed a {mover.opponent.value} flat on {move_text} for {whom}."
        else:
            text = f"{agent.name} played {move_text}."
        saved = self.recorder.after_move(self.game)
        if saved:
            return saved[0], f"{text} {saved[1]}"
        call = threat_call(self.game, mover)
        return (GOOD, f"{text} {call}") if call else (INFO, text)

    def play_text(self, text: str) -> Feedback | None:
        """Play a move typed in PTN."""
        try:
            move = parse_move(text)
        except NotationError as exc:
            return BAD, str(exc)
        return self.play(move)

    def confirm(self) -> Feedback | None:
        """Play the move waiting for confirmation, if there is one."""
        if self.pending is None:
            return None
        move, self.source = self.pending, None
        return self.play(move)

    def cancel_pending(self) -> bool:
        had = self.pending is not None
        self.pending = None
        return had
