"""Move types and legal move generation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tak.engine.board import Board, Direction, Square
from tak.engine.pieces import Piece, PieceType
from tak.engine.rules import drop_problem

if TYPE_CHECKING:
    from tak.engine.game import Game


@dataclass(frozen=True)
class PlaceMove:
    """Place a piece from reserve onto an empty square."""

    square: Square
    piece: PieceType = PieceType.FLAT


@dataclass(frozen=True)
class SlideMove:
    """Pick up part of a stack and carry it in a straight line.

    ``drops[i]`` is how many stones are left on the i-th square entered. The
    first drop uses the bottom of the carried stones, so the carried stack's
    top stone always ends up on the last square.
    """

    start: Square
    direction: Direction
    drops: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.drops or any(drop < 1 for drop in self.drops):
            raise ValueError("a slide must drop at least one stone on every square it enters")

    @property
    def count(self) -> int:
        """How many stones are picked up."""
        return sum(self.drops)


Move = PlaceMove | SlideMove


def legal_moves(game: Game) -> list[Move]:
    """Every legal move for the player to move (empty once the game is over)."""
    if game.is_over:
        return []
    moves: list[Move] = list(_placements(game))
    if not game.in_opening:
        moves.extend(_slides(game))
    return moves


def _placements(game: Game) -> Iterator[PlaceMove]:
    if game.in_opening:
        # Each player's first move places a flat stone of the opponent's color.
        types = [PieceType.FLAT]
    else:
        reserve = game.reserve(game.to_move)
        types = []
        if reserve.stones > 0:
            types += [PieceType.FLAT, PieceType.WALL]
        if reserve.caps > 0:
            types.append(PieceType.CAP)
    for square in game.board.squares():
        if not game.board.stack(square):
            for piece in types:
                yield PlaceMove(square, piece)


def _slides(game: Game) -> Iterator[SlideMove]:
    board = game.board
    color = game.to_move
    for square in board.squares():
        stack = board.stack(square)
        if not stack or stack[-1].color is not color:
            continue
        for direction in Direction:
            for count in range(1, min(len(stack), board.size) + 1):
                carried = stack[-count:]
                yield from _extend(board, square, direction, carried, 0, square, ())


def _extend(
    board: Board,
    origin: Square,
    direction: Direction,
    carried: tuple[Piece, ...],
    used: int,
    position: Square,
    drops: tuple[int, ...],
) -> Iterator[SlideMove]:
    """Choose how many stones to drop on the next square, recursively."""
    square = position.step(direction)
    if not board.in_bounds(square):
        return
    destination_top = board.top(square)
    remaining = len(carried) - used
    for take in range(1, remaining + 1):
        is_last = take == remaining
        dropped = carried[used : used + take]
        if drop_problem(square, destination_top, dropped, is_last) is not None:
            continue
        if is_last:
            yield SlideMove(origin, direction, (*drops, take))
        else:
            yield from _extend(
                board, origin, direction, carried, used + take, square, (*drops, take)
            )
