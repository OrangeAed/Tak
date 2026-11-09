"""Colors, piece types, and pieces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    """The two players. White is Player 1 and moves first."""

    WHITE = "white"
    BLACK = "black"

    @property
    def opponent(self) -> Color:
        return Color.BLACK if self is Color.WHITE else Color.WHITE

    def __str__(self) -> str:
        return self.value.capitalize()


class PieceType(Enum):
    """The three kinds of piece. The value is the PTN letter."""

    FLAT = "F"
    WALL = "S"  # standing stone
    CAP = "C"  # capstone


@dataclass(frozen=True)
class Piece:
    color: Color
    type: PieceType

    @property
    def counts_for_roads(self) -> bool:
        """Flat stones and capstones can be part of a road; walls cannot."""
        return self.type is not PieceType.WALL
