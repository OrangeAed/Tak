"""The Tak rules engine: pure game logic with no input or output."""

from tak.engine.board import BOARD_CONFIG, Board, Direction, Square
from tak.engine.game import Game, PlayedMove, Reserve
from tak.engine.moves import Move, PlaceMove, SlideMove, legal_moves
from tak.engine.notation import (
    NotationError,
    format_move,
    game_from_ptn,
    game_to_ptn,
    parse_move,
    parse_ptn,
)
from tak.engine.pieces import Color, Piece, PieceType
from tak.engine.rules import (
    GameOverError,
    GameResult,
    IllegalMoveError,
    Reason,
    count_flats,
    find_road,
)

__all__ = [
    "BOARD_CONFIG",
    "Board",
    "Color",
    "Direction",
    "Game",
    "GameOverError",
    "GameResult",
    "IllegalMoveError",
    "Move",
    "NotationError",
    "Piece",
    "PieceType",
    "PlaceMove",
    "PlayedMove",
    "Reason",
    "Reserve",
    "SlideMove",
    "Square",
    "count_flats",
    "find_road",
    "format_move",
    "game_from_ptn",
    "game_to_ptn",
    "legal_moves",
    "parse_move",
    "parse_ptn",
]
