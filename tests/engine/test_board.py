"""Tests for squares, directions, and the board."""

from __future__ import annotations

import pytest

from tak.engine import BOARD_CONFIG, Board, Color, Direction, Piece, PieceType, Square

WHITE_FLAT = Piece(Color.WHITE, PieceType.FLAT)
BLACK_WALL = Piece(Color.BLACK, PieceType.WALL)


def test_piece_supply_matches_the_rules_table() -> None:
    assert BOARD_CONFIG == {
        3: (10, 0),
        4: (15, 0),
        5: (21, 1),
        6: (30, 1),
        7: (40, 2),
        8: (50, 2),
    }


@pytest.mark.parametrize(
    ("name", "file_rank"),
    [("a1", (0, 0)), ("e5", (4, 4)), ("h8", (7, 7)), ("c4", (2, 3))],
)
def test_square_names_round_trip(name: str, file_rank: tuple[int, int]) -> None:
    assert Square.from_name(name) == file_rank
    assert str(Square(*file_rank)) == name


@pytest.mark.parametrize("bad", ["", "a", "a0", "a9", "i1", "A1", "a11", "1a"])
def test_bad_square_names_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid square"):
        Square.from_name(bad)


def test_stepping_moves_in_each_direction() -> None:
    center = Square(2, 2)
    assert center.step(Direction.LEFT) == Square(1, 2)
    assert center.step(Direction.RIGHT) == Square(3, 2)
    assert center.step(Direction.UP) == Square(2, 3)
    assert center.step(Direction.DOWN) == Square(2, 1)
    assert center.step(Direction.RIGHT, 2) == Square(4, 2)


def test_directions_use_ptn_symbols() -> None:
    assert Direction("<") is Direction.LEFT
    assert Direction(">") is Direction.RIGHT
    assert Direction("+") is Direction.UP
    assert Direction("-") is Direction.DOWN


def test_color_opponent_and_names() -> None:
    assert Color.WHITE.opponent is Color.BLACK
    assert Color.BLACK.opponent is Color.WHITE
    assert str(Color.WHITE) == "White"


def test_only_walls_are_excluded_from_roads() -> None:
    assert WHITE_FLAT.counts_for_roads
    assert Piece(Color.WHITE, PieceType.CAP).counts_for_roads
    assert not BLACK_WALL.counts_for_roads


class TestBoard:
    @pytest.mark.parametrize("size", [2, 9, 0, -1])
    def test_unsupported_sizes_are_rejected(self, size: int) -> None:
        with pytest.raises(ValueError, match="unsupported board size"):
            Board(size)

    @pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
    def test_new_board_is_empty(self, size: int) -> None:
        board = Board(size)
        assert len(list(board.squares())) == size * size
        assert all(board.stack(square) == () for square in board.squares())
        assert not board.is_full()

    def test_bounds(self) -> None:
        board = Board(5)
        assert board.in_bounds(Square(0, 0))
        assert board.in_bounds(Square(4, 4))
        assert not board.in_bounds(Square(5, 0))
        assert not board.in_bounds(Square(0, 5))
        assert not board.in_bounds(Square(-1, 0))

    def test_off_board_access_raises_instead_of_wrapping(self) -> None:
        board = Board(5)
        with pytest.raises(IndexError):
            board.stack(Square(-1, 0))
        with pytest.raises(IndexError):
            board.set_stack(Square(0, 5), (WHITE_FLAT,))

    def test_stacks_and_tops(self) -> None:
        board = Board(5)
        square = Square(2, 2)
        assert board.top(square) is None
        board.set_stack(square, (WHITE_FLAT, BLACK_WALL))
        assert board.stack(square) == (WHITE_FLAT, BLACK_WALL)
        assert board.top(square) == BLACK_WALL

    def test_full_board(self) -> None:
        board = Board(3)
        squares = list(board.squares())
        for square in squares[:-1]:
            board.set_stack(square, (WHITE_FLAT,))
        assert not board.is_full()
        board.set_stack(squares[-1], (WHITE_FLAT,))
        assert board.is_full()

    def test_copy_is_independent(self) -> None:
        board = Board(3)
        clone = board.copy()
        clone.set_stack(Square(0, 0), (WHITE_FLAT,))
        assert board.stack(Square(0, 0)) == ()
        assert clone.stack(Square(0, 0)) == (WHITE_FLAT,)
