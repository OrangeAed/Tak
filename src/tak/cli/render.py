"""Turn game state into text for the terminal."""

from __future__ import annotations

from tak.engine.board import BOARD_CONFIG, Board, Square
from tak.engine.game import Game
from tak.engine.pieces import Color, Piece, PieceType
from tak.engine.rules import GameResult, Reason, count_flats, find_road

LEGEND = (
    "Pieces: w = white flat, b = black flat, Sw/Sb = wall, Cw/Cb = capstone. "
    "Stacks read bottom to top, so the rightmost piece is on top."
)

_TYPE_PREFIX = {PieceType.FLAT: "", PieceType.WALL: "S", PieceType.CAP: "C"}


def piece_label(piece: Piece) -> str:
    return _TYPE_PREFIX[piece.type] + ("w" if piece.color is Color.WHITE else "b")


def stack_label(stack: tuple[Piece, ...]) -> str:
    return "".join(piece_label(piece) for piece in stack) or "."


def render_board(board: Board) -> str:
    """Draw the board with rank 1 at the bottom and file a on the left."""
    size = board.size
    labels = {square: stack_label(board.stack(square)) for square in board.squares()}
    width = max(4, max(len(label) for label in labels.values()) + 2)
    rule = "   +" + "+".join("-" * width for _ in range(size)) + "+"

    lines = []
    for rank in reversed(range(size)):
        lines.append(rule)
        cells = "|".join(labels[Square(file, rank)].center(width) for file in range(size))
        lines.append(f"{rank + 1:>2} |{cells}|")
    lines.append(rule)
    lines.append("    " + " ".join(chr(ord("a") + file).center(width) for file in range(size)))
    return "\n".join(lines)


def _describe_reserve(game: Game, color: Color) -> str:
    reserve = game.reserve(color)
    text = f"{color}: {reserve.stones} stones"
    if BOARD_CONFIG[game.size][1]:
        text += f", {reserve.caps} capstone" + ("" if reserve.caps == 1 else "s")
    return text


def render_status(game: Game) -> str:
    """Reserves and whose turn it is."""
    reserves = "   ".join(_describe_reserve(game, color) for color in Color)
    return f"{reserves}\nTurn {game.turn_number}: {game.to_move} to move."


def describe_result(game: Game, result: GameResult) -> str:
    """A sentence saying who won and how."""
    if result.reason is Reason.ROAD:
        assert result.winner is not None
        road = find_road(game.board, result.winner) or []
        route = " -> ".join(str(square) for square in road)
        return f"{result.winner} wins with a road: {route}"
    white = count_flats(game.board, Color.WHITE)
    black = count_flats(game.board, Color.BLACK)
    if result.winner is None:
        return f"Draw: both players have {white} flat stones on top."
    return f"{result.winner} wins on flat stones, {white} to {black} (White to Black)."
