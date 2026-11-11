"""Click to move: turning clicks on squares into Tak moves.

This is pure logic with no drawing, so it can be tested on its own. Every step
is checked against the engine's list of legal moves, so a click can only ever
lead toward a move the engine would accept.

How a move is made with clicks:

- **Place:** click an empty square. A flat stone is placed unless a wall or
  capstone was chosen first (the window offers that in its panel, and with
  Shift+click and Ctrl+click).
- **Move a stack:** click a stack you control to lift it. As many stones as
  the carry limit allows are lifted; click the stack again to carry one fewer
  (down to none, which puts it back). Then click a neighbouring square to
  drop one stone there, click that square again to drop another there, or
  click the next square along to move on. The move is played as soon as the
  last carried stone is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tak.engine.board import Board, Direction, Square
from tak.engine.game import Game
from tak.engine.moves import Move, PlaceMove, SlideMove
from tak.engine.pieces import Piece, PieceType
from tak.engine.rules import IllegalMoveError


@dataclass(frozen=True)
class Lift:
    """A stack move in progress: stones picked up and maybe some already dropped."""

    start: Square
    count: int  # stones picked up
    direction: Direction | None = None  # set by the first drop
    drops: tuple[int, ...] = ()  # stones dropped on each square entered so far

    @property
    def remaining(self) -> int:
        """Stones still being carried."""
        return self.count - sum(self.drops)

    @property
    def position(self) -> Square:
        """The square the carried stones are over: the start, or the last drop."""
        if self.direction is None:
            return self.start
        return self.start.step(self.direction, len(self.drops))

    def could_become(self, move: SlideMove) -> bool:
        """True if dropping the rest of the stones could turn this into ``move``."""
        if move.start != self.start or move.count != self.count:
            return False
        if self.direction is None:
            return True
        if move.direction is not self.direction or len(move.drops) < len(self.drops):
            return False
        last = len(self.drops) - 1
        return move.drops[:last] == self.drops[:last] and move.drops[last] >= self.drops[last]


@dataclass(frozen=True)
class Outcome:
    """What a click did: a finished ``move`` to play, or a ``message`` to show (or neither)."""

    move: Move | None = None
    message: str | None = None
    error: bool = False


class MoveBuilder:
    """Collects clicks into a move for the player whose turn it is."""

    def __init__(self) -> None:
        self.lift: Lift | None = None

    def cancel(self) -> bool:
        """Drop any stack move in progress; returns True if there was one."""
        had_lift = self.lift is not None
        self.lift = None
        return had_lift

    # --- Clicking ---------------------------------------------------------------

    def click(self, game: Game, square: Square, piece: PieceType = PieceType.FLAT) -> Outcome:
        """Handle a click on ``square``; ``piece`` is what to place on an empty square."""
        if game.is_over:
            return Outcome(message="The game is over.", error=True)
        if self.lift is None:
            return self._click_idle(game, square, piece)
        return self._click_lifted(game, self.lift, square)

    def _click_idle(self, game: Game, square: Square, piece: PieceType) -> Outcome:
        stack = game.board.stack(square)
        if not stack:
            if game.in_opening:
                piece = PieceType.FLAT  # the opening only allows a flat, so don't make them choose
            move = PlaceMove(square, piece)
            try:
                game.check(move)
            except IllegalMoveError as exc:
                return Outcome(message=f"Can't place there: {exc}.", error=True)
            return Outcome(move=move)
        return self._lift(game, square)

    def _lift(self, game: Game, square: Square) -> Outcome:
        if game.in_opening:
            return Outcome(
                message="On the first turn you place a flat stone; stacks can't be moved yet.",
                error=True,
            )
        top = game.board.stack(square)[-1]
        if top.color is not game.to_move:
            return Outcome(message=f"The stack on {square} isn't yours to move.", error=True)
        counts = [move.count for move in self._slides(game) if move.start == square]
        if not counts:
            return Outcome(message=f"The stack on {square} has nowhere to go.", error=True)
        self.lift = Lift(square, max(counts))
        return Outcome(message=self.describe())

    def _click_lifted(self, game: Game, lift: Lift, square: Square) -> Outcome:
        candidate = self._next(lift, square)
        if candidate is None:
            # Clicking another of your own stacks before dropping anything switches to it.
            stack = game.board.stack(square)
            if lift.direction is None and stack and stack[-1].color is game.to_move:
                self.lift = None
                return self._lift(game, square)
            return Outcome(message=self._hint(lift), error=True)
        if candidate.count == 0:
            self.lift = None
            return Outcome(message=f"Put the stack on {lift.start} back.")

        matches = [move for move in self._slides(game) if candidate.could_become(move)]
        if not matches:
            return Outcome(message=f"Can't drop there. {self._hint(lift)}", error=True)
        if candidate.remaining == 0:
            assert candidate.direction is not None
            self.lift = None
            return Outcome(move=SlideMove(candidate.start, candidate.direction, candidate.drops))
        self.lift = candidate
        return Outcome(message=self.describe())

    @staticmethod
    def _next(lift: Lift, square: Square) -> Lift | None:
        """The lift after clicking ``square``, or ``None`` if that click means nothing."""
        if lift.direction is None:
            if square == lift.start:
                return replace(lift, count=lift.count - 1)  # carry one fewer
            for direction in Direction:
                if lift.start.step(direction) == square:
                    return replace(lift, direction=direction, drops=(1,))
            return None
        if square == lift.position:  # drop one more here
            return replace(lift, drops=(*lift.drops[:-1], lift.drops[-1] + 1))
        if square == lift.position.step(lift.direction):  # move on and drop one
            return replace(lift, drops=(*lift.drops, 1))
        return None

    @staticmethod
    def _slides(game: Game) -> list[SlideMove]:
        return [move for move in game.legal_moves() if isinstance(move, SlideMove)]

    # --- Describing and previewing ----------------------------------------------

    def describe(self) -> str | None:
        """A line for the panel saying what the player is doing and can do next."""
        lift = self.lift
        if lift is None:
            return None
        stones = "stone" if lift.remaining == 1 else "stones"
        if lift.direction is None:
            return (
                f"Carrying {lift.count} {stones} from {lift.start}. Click a neighbouring "
                f"square to move, or {lift.start} again to carry fewer."
            )
        ahead = lift.position.step(lift.direction)
        return (
            f"Carrying {lift.remaining} more {stones}. Click {lift.position} to drop one "
            f"there, or {ahead} to move on."
        )

    @staticmethod
    def _hint(lift: Lift) -> str:
        if lift.direction is None:
            return f"Click a square next to {lift.start}, or {lift.start} to carry fewer."
        ahead = lift.position.step(lift.direction)
        return f"Click {lift.position} or {ahead}, or right-click to cancel."

    def targets(self, game: Game) -> set[Square]:
        """Squares where a click would do something useful with the current lift."""
        lift = self.lift
        if lift is None or game.is_over:
            return set()
        slides = [move for move in self._slides(game) if lift.could_become(move)]
        squares = set()
        for square in game.board.squares():
            candidate = self._next(lift, square)
            if candidate is None:
                continue
            if candidate.count == 0 or any(candidate.could_become(m) for m in slides):
                squares.add(square)
        return squares

    def preview(self, board: Board) -> tuple[Board, tuple[Square, tuple[Piece, ...]] | None]:
        """The board as it looks mid-move, and the stones still in the air (and over where).

        Without a lift this is just ``board`` and nothing in the air.
        """
        lift = self.lift
        if lift is None:
            return board, None
        shown = board.copy()
        stack = board.stack(lift.start)
        carried = stack[-lift.count :]
        shown.set_stack(lift.start, stack[: -lift.count])
        used = 0
        square = lift.start
        if lift.direction is not None:
            for drop in lift.drops:
                square = square.step(lift.direction)
                shown.set_stack(square, shown.stack(square) + carried[used : used + drop])
                used += drop
        return shown, (lift.position, carried[used:])


def touched_squares(move: Move) -> list[Square]:
    """Every square a move changed: where a piece was placed, or a slide's whole path."""
    if isinstance(move, PlaceMove):
        return [move.square]
    return [move.start.step(move.direction, i) for i in range(len(move.drops) + 1)]
