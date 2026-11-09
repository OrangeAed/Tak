"""The board: squares, directions, stacks, and the per-size piece supply."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from typing import NamedTuple

from tak.engine.pieces import Piece

#: Board size -> (stones per player, capstones per player).
BOARD_CONFIG: dict[int, tuple[int, int]] = {
    3: (10, 0),
    4: (15, 0),
    5: (21, 1),
    6: (30, 1),
    7: (40, 2),
    8: (50, 2),
}


class Direction(Enum):
    """A direction of travel. The value is the PTN symbol.

    Directions are from White's point of view: ``<`` is toward the a-file,
    ``-`` is toward rank 1.
    """

    LEFT = "<"
    RIGHT = ">"
    UP = "+"
    DOWN = "-"

    @property
    def delta(self) -> tuple[int, int]:
        """The (file, rank) change for one step in this direction."""
        return _DELTAS[self]


_DELTAS: dict[Direction, tuple[int, int]] = {
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
    Direction.UP: (0, 1),
    Direction.DOWN: (0, -1),
}


class Square(NamedTuple):
    """A board square as zero-based (file, rank); ``a1`` is ``Square(0, 0)``."""

    file: int
    rank: int

    @property
    def name(self) -> str:
        return f"{chr(ord('a') + self.file)}{self.rank + 1}"

    @classmethod
    def from_name(cls, name: str) -> Square:
        if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
            raise ValueError(f"invalid square name: {name!r}")
        return cls(ord(name[0]) - ord("a"), int(name[1]) - 1)

    def step(self, direction: Direction, distance: int = 1) -> Square:
        d_file, d_rank = direction.delta
        return Square(self.file + d_file * distance, self.rank + d_rank * distance)

    def __str__(self) -> str:
        return self.name


class Board:
    """A square grid of stacks. Each stack is a tuple of pieces, bottom first."""

    def __init__(self, size: int) -> None:
        if size not in BOARD_CONFIG:
            valid = ", ".join(str(s) for s in BOARD_CONFIG)
            raise ValueError(f"unsupported board size {size}; choose one of {valid}")
        self.size = size
        self._cells: list[tuple[Piece, ...]] = [()] * (size * size)

    def in_bounds(self, square: Square) -> bool:
        return 0 <= square.file < self.size and 0 <= square.rank < self.size

    def stack(self, square: Square) -> tuple[Piece, ...]:
        if not self.in_bounds(square):
            raise IndexError(f"{square} is off the board")
        return self._cells[square.rank * self.size + square.file]

    def top(self, square: Square) -> Piece | None:
        stack = self.stack(square)
        return stack[-1] if stack else None

    def set_stack(self, square: Square, stack: tuple[Piece, ...]) -> None:
        if not self.in_bounds(square):
            raise IndexError(f"{square} is off the board")
        self._cells[square.rank * self.size + square.file] = stack

    def squares(self) -> Iterator[Square]:
        """Every square, rank by rank from a1."""
        for rank in range(self.size):
            for file in range(self.size):
                yield Square(file, rank)

    def is_full(self) -> bool:
        return all(self._cells)

    def copy(self) -> Board:
        clone = Board(self.size)
        clone._cells = list(self._cells)
        return clone
