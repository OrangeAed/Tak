"""Tests for turning game state into text."""

from __future__ import annotations

from collections.abc import Callable

from tak.cli.render import (
    describe_result,
    piece_label,
    render_board,
    render_status,
    stack_label,
)
from tak.engine import Board, Color, Game, Piece, PieceType, Square


def piece(color: Color, kind: PieceType) -> Piece:
    return Piece(color, kind)


def test_piece_labels() -> None:
    assert piece_label(piece(Color.WHITE, PieceType.FLAT)) == "w"
    assert piece_label(piece(Color.BLACK, PieceType.FLAT)) == "b"
    assert piece_label(piece(Color.WHITE, PieceType.WALL)) == "Sw"
    assert piece_label(piece(Color.BLACK, PieceType.CAP)) == "Cb"


def test_stack_labels_read_bottom_to_top() -> None:
    stack = (
        piece(Color.BLACK, PieceType.FLAT),
        piece(Color.WHITE, PieceType.FLAT),
        piece(Color.BLACK, PieceType.WALL),
    )
    assert stack_label(stack) == "bwSb"
    assert stack_label(()) == "."


def test_empty_board_layout() -> None:
    lines = render_board(Board(3)).splitlines()
    assert len(lines) == 3 * 2 + 1 + 1  # a rule above each rank, one below, and file letters
    assert lines[0].strip().startswith("+")
    assert lines[1].startswith(" 3 |")  # the top rank is printed first
    assert lines[-3].startswith(" 1 |")
    assert lines[-1].split() == ["a", "b", "c"]


def test_pieces_appear_on_the_right_squares() -> None:
    board = Board(3)
    board.set_stack(Square.from_name("a1"), (piece(Color.BLACK, PieceType.FLAT),))
    board.set_stack(Square.from_name("c3"), (piece(Color.WHITE, PieceType.CAP),))
    lines = render_board(board).splitlines()
    top_rank, bottom_rank = lines[1], lines[-3]
    assert top_rank.split("|")[3].strip() == "Cw"  # c3: third cell of the top rank
    assert bottom_rank.split("|")[1].strip() == "b"  # a1: first cell of the bottom rank


def test_columns_widen_for_tall_stacks_and_stay_aligned() -> None:
    board = Board(3)
    board.set_stack(Square.from_name("b2"), (piece(Color.WHITE, PieceType.FLAT),) * 7)
    lines = render_board(board).splitlines()
    assert "wwwwwww" in render_board(board)
    assert len({len(line) for line in lines[:-1]}) == 1  # every grid line is the same width


def test_status_shows_reserves_and_whose_turn() -> None:
    status = render_status(Game(5))
    assert "White: 21 stones, 1 capstone" in status
    assert "Black: 21 stones, 1 capstone" in status
    assert "Turn 1: White to move." in status


def test_status_omits_capstones_on_small_boards() -> None:
    status = render_status(Game(3))
    assert "capstone" not in status
    assert "White: 10 stones" in status


def test_status_announces_tak_just_before_the_turn_line() -> None:
    lines = render_status(Game(3), tak=True).splitlines()
    assert lines[1:] == ["Tak!", "Turn 1: White to move."]
    assert "Tak!" not in render_status(Game(3))


def test_status_tracks_turns(started: Callable[..., Game]) -> None:
    status = render_status(started("a1 e5 c3"))
    assert "Turn 2: Black to move." in status
    assert "White: 19 stones" in status


def test_describe_road_win(started: Callable[..., Game]) -> None:
    game = started("a1 a2 b2 a3 c2", size=3)
    assert game.result is not None
    assert describe_result(game, game.result) == "White wins with a road: a2 -> b2 -> c2"


def test_describe_flat_win(started: Callable[..., Game]) -> None:
    game = started("a2 a3 c3 Sb1 Sb2 b3 a1 c2 c1", size=3)
    assert game.result is not None
    assert describe_result(game, game.result) == (
        "White wins on flat stones, 4 to 3 (White to Black)."
    )


def test_describe_draw(started: Callable[..., Game]) -> None:
    game = started("a2 a3 c3 Sb1 Sb2 b3 Sa1 c2 c1", size=3)
    assert game.result is not None
    assert describe_result(game, game.result) == "Draw: both players have 3 flat stones on top."
