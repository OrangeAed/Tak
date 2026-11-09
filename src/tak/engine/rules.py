"""The rules of Tak: move validation, road detection, and win conditions.

Everything here works on a :class:`Board` alone, so it can be tested without
playing a whole game.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from tak.engine.board import Board, Direction, Square
from tak.engine.pieces import Color, Piece, PieceType

if TYPE_CHECKING:
    from tak.engine.moves import SlideMove


class IllegalMoveError(ValueError):
    """Raised when a move breaks the rules. The message says why."""


class GameOverError(IllegalMoveError):
    """Raised when a move is attempted after the game has ended."""


class Reason(Enum):
    ROAD = "road"
    FLAT = "flat"


@dataclass(frozen=True)
class GameResult:
    """How a finished game ended. ``winner`` is ``None`` for a draw."""

    winner: Color | None
    reason: Reason

    @property
    def ptn(self) -> str:
        """The result token used in PTN, e.g. ``R-0`` or ``1/2-1/2``."""
        if self.winner is None:
            return "1/2-1/2"
        letter = "R" if self.reason is Reason.ROAD else "F"
        return f"{letter}-0" if self.winner is Color.WHITE else f"0-{letter}"


def _stones(count: int) -> str:
    return f"{count} stone" if count == 1 else f"{count} stones"


# --- Moving stacks -----------------------------------------------------------


def drop_problem(
    square: Square, destination_top: Piece | None, dropped: Sequence[Piece], is_last: bool
) -> str | None:
    """Why ``dropped`` can't be left on ``square``, or ``None`` if it can.

    ``destination_top`` is the piece currently on top of that square. Walls and
    capstones can't be stacked on, except that a capstone moving *alone* as the
    final drop of a move flattens a wall.
    """
    if destination_top is None or destination_top.type is PieceType.FLAT:
        return None
    if destination_top.type is PieceType.CAP:
        return f"blocked by the capstone on {square}"
    if not is_last:
        return f"blocked by the wall on {square}"
    if len(dropped) != 1 or dropped[0].type is not PieceType.CAP:
        return f"the wall on {square} can only be flattened by a capstone moving onto it alone"
    return None


def slide_problem(board: Board, color: Color, move: SlideMove) -> str | None:
    """Why ``color`` can't make this slide on ``board``, or ``None`` if it's legal."""
    start = move.start
    if not board.in_bounds(start):
        return f"{start} is not on the board"
    stack = board.stack(start)
    if not stack:
        return f"there are no stones on {start}"
    if stack[-1].color is not color:
        return f"you don't control the stack on {start} (its top piece is {stack[-1].color.value})"
    count = move.count
    if count > len(stack):
        return f"{start} has only {_stones(len(stack))}, so you can't pick up {count}"
    if count > board.size:
        return (
            f"you can't pick up {count} stones: the carry limit on a "
            f"{board.size}x{board.size} board is {board.size}"
        )

    carried = stack[-count:]
    used = 0
    last = len(move.drops) - 1
    for index, take in enumerate(move.drops):
        square = move.start.step(move.direction, index + 1)
        if not board.in_bounds(square):
            return "the move runs off the edge of the board"
        dropped = carried[used : used + take]
        used += take
        problem = drop_problem(square, board.top(square), dropped, index == last)
        if problem is not None:
            return problem
    return None


# --- Roads -------------------------------------------------------------------


def _is_road_square(board: Board, color: Color, square: Square) -> bool:
    top = board.top(square)
    return top is not None and top.color is color and top.counts_for_roads


def _find_road_along(board: Board, color: Color, axis: int) -> list[Square] | None:
    """Shortest road for ``color`` from one edge to the opposite edge.

    ``axis`` 0 joins the a-file to the last file; axis 1 joins rank 1 to the
    last rank.
    """
    last = board.size - 1
    parent: dict[Square, Square | None] = {}
    queue: deque[Square] = deque()
    for index in range(board.size):  # the starting edge, in the same order as board.squares()
        square = Square(0, index) if axis == 0 else Square(index, 0)
        if _is_road_square(board, color, square):
            parent[square] = None
            queue.append(square)
    while queue:
        square = queue.popleft()
        if square[axis] == last:
            path: list[Square] = []
            node: Square | None = square
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        for direction in Direction:
            neighbor = square.step(direction)
            if (
                board.in_bounds(neighbor)
                and neighbor not in parent
                and _is_road_square(board, color, neighbor)
            ):
                parent[neighbor] = square
                queue.append(neighbor)
    return None


def find_road(board: Board, color: Color) -> list[Square] | None:
    """A shortest road for ``color``, as a list of squares, or ``None``.

    A road is an orthogonally connected chain of squares whose top piece is
    one of ``color``'s flat stones or capstones, joining opposite edges.
    """
    return _find_road_along(board, color, 0) or _find_road_along(board, color, 1)


# --- Ending the game ---------------------------------------------------------


def count_flats(board: Board, color: Color) -> int:
    """Flat stones of ``color`` on top of stacks.

    Walls, capstones, and flats covered by another piece don't count.
    """
    total = 0
    for square in board.squares():
        top = board.top(square)
        if top is not None and top.color is color and top.type is PieceType.FLAT:
            total += 1
    return total


def end_of_game(board: Board, mover: Color, out_of_pieces: bool) -> GameResult | None:
    """The result after ``mover`` has just moved, or ``None`` if play continues.

    A road ends the game immediately. If both players have a road, the player
    who just moved wins. Otherwise the game ends when the board is full or a
    player has placed their last piece, and the most flat stones on top wins.
    """
    for color in (mover, mover.opponent):
        if find_road(board, color) is not None:
            return GameResult(color, Reason.ROAD)
    if out_of_pieces or board.is_full():
        white = count_flats(board, Color.WHITE)
        black = count_flats(board, Color.BLACK)
        if white == black:
            return GameResult(None, Reason.FLAT)
        return GameResult(Color.WHITE if white > black else Color.BLACK, Reason.FLAT)
    return None
