"""Tests for the board-level rules: slides, roads, flats, and end of game."""

from __future__ import annotations

import re

import pytest

from tak.engine import (
    Board,
    Color,
    Direction,
    GameResult,
    Piece,
    PieceType,
    Reason,
    SlideMove,
    Square,
    count_flats,
    find_road,
)
from tak.engine.rules import drop_problem, end_of_game, slide_problem

_TOKEN = re.compile(r"[SC]?[wb]")
_TYPES = {"": PieceType.FLAT, "S": PieceType.WALL, "C": PieceType.CAP}

WHITE, BLACK = Color.WHITE, Color.BLACK


def stack_of(text: str) -> tuple[Piece, ...]:
    """``"wbSw"`` -> white flat, black flat, white wall (bottom to top)."""
    pieces = []
    for token in _TOKEN.findall(text):
        color = WHITE if token[-1] == "w" else BLACK
        pieces.append(Piece(color, _TYPES[token[:-1]]))
    return tuple(pieces)


def board_with(size: int, layout: dict[str, str]) -> Board:
    board = Board(size)
    for name, text in layout.items():
        board.set_stack(Square.from_name(name), stack_of(text))
    return board


def slide(start: str, symbol: str, *drops: int) -> SlideMove:
    return SlideMove(Square.from_name(start), Direction(symbol), drops)


def road_names(board: Board, color: Color) -> list[str] | None:
    road = find_road(board, color)
    return None if road is None else [str(square) for square in road]


# --- Roads -------------------------------------------------------------------


class TestRoads:
    def test_empty_board_has_no_road(self) -> None:
        board = Board(5)
        assert find_road(board, WHITE) is None
        assert find_road(board, BLACK) is None

    def test_straight_road_across(self) -> None:
        board = board_with(5, {f"{file}3": "w" for file in "abcde"})
        assert road_names(board, WHITE) == ["a3", "b3", "c3", "d3", "e3"]
        assert find_road(board, BLACK) is None

    def test_straight_road_top_to_bottom(self) -> None:
        board = board_with(5, {f"c{rank}": "b" for rank in range(1, 6)})
        assert road_names(board, BLACK) == ["c1", "c2", "c3", "c4", "c5"]
        assert find_road(board, WHITE) is None

    def test_road_may_turn(self) -> None:
        board = board_with(5, {name: "w" for name in ["a1", "b1", "b2", "b3", "c3", "d3", "e3"]})
        assert road_names(board, WHITE) == ["a1", "b1", "b2", "b3", "c3", "d3", "e3"]

    def test_road_may_use_the_far_edges(self) -> None:
        board = board_with(5, {name: "w" for name in ["e1", "e2", "e3", "e4", "e5"]})
        assert find_road(board, WHITE) is not None

    def test_a_gap_breaks_the_road(self) -> None:
        board = board_with(5, {f"{file}3": "w" for file in "abce"})
        assert find_road(board, WHITE) is None

    def test_diagonals_do_not_connect(self) -> None:
        board = board_with(5, {name: "w" for name in ["a1", "b2", "c3", "d4", "e5"]})
        assert find_road(board, WHITE) is None

    def test_road_must_reach_opposite_edges(self) -> None:
        board = board_with(5, {f"{file}3": "w" for file in "abcd"})
        assert find_road(board, WHITE) is None

    def test_a_wall_breaks_the_road(self) -> None:
        layout = {f"{file}3": "w" for file in "abcde"}
        layout["c3"] = "Sw"
        assert find_road(board_with(5, layout), WHITE) is None

    def test_a_capstone_is_part_of_a_road(self) -> None:
        layout = {f"{file}3": "w" for file in "abcde"}
        layout["c3"] = "Cw"
        assert road_names(board_with(5, layout), WHITE) is not None

    def test_an_opposing_piece_breaks_the_road(self) -> None:
        layout = {f"{file}3": "w" for file in "abcde"}
        layout["c3"] = "b"
        board = board_with(5, layout)
        assert find_road(board, WHITE) is None
        assert find_road(board, BLACK) is None

    def test_only_the_top_of_a_stack_counts(self) -> None:
        layout = {f"{file}3": "w" for file in "abcde"}
        layout["c3"] = "bw"  # white on top of black
        assert find_road(board_with(5, layout), WHITE) is not None
        layout["c3"] = "wb"  # black on top of white
        assert find_road(board_with(5, layout), WHITE) is None

    def test_roads_work_on_every_board_size(self) -> None:
        for size in range(3, 9):
            files = "abcdefgh"[:size]
            board = board_with(size, {f"{file}1": "w" for file in files})
            assert find_road(board, WHITE) is not None, size

    def test_finds_a_shortest_road(self) -> None:
        # A full board of white flats: the shortest road is a straight line.
        board = board_with(5, {f"{f}{r}": "w" for f in "abcde" for r in range(1, 6)})
        road = find_road(board, WHITE)
        assert road is not None
        assert len(road) == 5


# --- Flat counting -----------------------------------------------------------


class TestFlats:
    def test_counts_flats_on_top_only(self) -> None:
        board = board_with(
            5,
            {
                "a1": "w",
                "a2": "w",
                "a3": "bw",  # covered black flat, white flat on top: counts for white
                "a4": "wb",  # white flat covered by black: counts for black
                "a5": "Sw",  # walls don't count
                "b1": "Cb",  # capstones don't count
                "b2": "b",
            },
        )
        assert count_flats(board, WHITE) == 3
        assert count_flats(board, BLACK) == 2

    def test_empty_board_counts_zero(self) -> None:
        assert count_flats(Board(5), WHITE) == 0


# --- Ending the game ---------------------------------------------------------


class TestEndOfGame:
    def test_play_continues_with_no_road_and_pieces_left(self) -> None:
        board = board_with(5, {"a1": "w", "b1": "b"})
        assert end_of_game(board, WHITE, out_of_pieces=False) is None

    def test_a_road_wins(self) -> None:
        board = board_with(3, {"a1": "w", "b1": "w", "c1": "w"})
        assert end_of_game(board, WHITE, False) == GameResult(WHITE, Reason.ROAD)

    def test_a_road_ends_the_game_even_if_the_other_player_moved(self) -> None:
        # The road can be revealed by the opponent's move uncovering a stack.
        board = board_with(3, {"a1": "b", "b1": "b", "c1": "b"})
        assert end_of_game(board, WHITE, False) == GameResult(BLACK, Reason.ROAD)

    def test_double_road_goes_to_the_player_who_just_moved(self) -> None:
        board = board_with(3, {"a1": "w", "b1": "w", "c1": "w", "a3": "b", "b3": "b", "c3": "b"})
        assert end_of_game(board, WHITE, False) == GameResult(WHITE, Reason.ROAD)
        assert end_of_game(board, BLACK, False) == GameResult(BLACK, Reason.ROAD)

    def test_a_road_beats_a_flat_count_when_pieces_run_out(self) -> None:
        board = board_with(3, {"a1": "b", "b1": "b", "c1": "b", "a3": "w", "b3": "w"})
        assert end_of_game(board, WHITE, out_of_pieces=True) == GameResult(BLACK, Reason.ROAD)

    def test_running_out_of_pieces_ends_on_flats(self) -> None:
        board = board_with(3, {"a1": "w", "c3": "w", "b2": "b"})
        assert end_of_game(board, WHITE, True) == GameResult(WHITE, Reason.FLAT)

    def test_flat_win_for_black(self) -> None:
        board = board_with(3, {"a1": "w", "c3": "b", "b2": "b"})
        assert end_of_game(board, WHITE, True) == GameResult(BLACK, Reason.FLAT)

    def test_a_tie_is_a_draw(self) -> None:
        board = board_with(3, {"a1": "w", "c3": "b"})
        assert end_of_game(board, WHITE, True) == GameResult(None, Reason.FLAT)

    def test_walls_and_capstones_do_not_count_toward_flats(self) -> None:
        board = board_with(3, {"a1": "w", "b1": "Sw", "c1": "Cw", "a3": "b"})
        assert end_of_game(board, BLACK, True) == GameResult(None, Reason.FLAT)

    def test_a_full_board_ends_the_game(self) -> None:
        # Walls keep either player from having a road.
        board = board_with(
            3,
            {
                "a3": "w", "b3": "b", "c3": "w",
                "a2": "b", "b2": "Sw", "c2": "b",
                "a1": "w", "b1": "Sb", "c1": "w",
            },
        )  # fmt: skip
        assert board.is_full()
        assert end_of_game(board, WHITE, out_of_pieces=False) == GameResult(WHITE, Reason.FLAT)

    @pytest.mark.parametrize(
        ("result", "token"),
        [
            (GameResult(WHITE, Reason.ROAD), "R-0"),
            (GameResult(BLACK, Reason.ROAD), "0-R"),
            (GameResult(WHITE, Reason.FLAT), "F-0"),
            (GameResult(BLACK, Reason.FLAT), "0-F"),
            (GameResult(None, Reason.FLAT), "1/2-1/2"),
        ],
    )
    def test_result_tokens(self, result: GameResult, token: str) -> None:
        assert result.ptn == token


# --- Slide rules -------------------------------------------------------------


class TestSlideProblems:
    def test_a_simple_slide_is_fine(self) -> None:
        board = board_with(5, {"a1": "w"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1)) is None

    def test_cannot_move_from_an_empty_square(self) -> None:
        problem = slide_problem(Board(5), WHITE, slide("a1", ">", 1))
        assert problem is not None and "no stones" in problem

    def test_cannot_move_a_stack_you_do_not_control(self) -> None:
        board = board_with(5, {"a1": "wb"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 1))
        assert problem is not None and "don't control" in problem

    def test_can_move_a_stack_with_your_piece_on_top_of_theirs(self) -> None:
        board = board_with(5, {"a1": "bw"})
        assert slide_problem(board, WHITE, slide("a1", ">", 2)) is None

    def test_cannot_pick_up_more_stones_than_are_there(self) -> None:
        board = board_with(5, {"a1": "ww"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 3))
        assert problem is not None and "only 2 stones" in problem

    def test_carry_limit_is_the_board_size(self) -> None:
        board = board_with(5, {"a1": "wwwwww"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 3, 3))
        assert problem is not None and "carry limit" in problem
        assert slide_problem(board, WHITE, slide("a1", ">", 3, 2)) is None  # picks up 5

    def test_carry_limit_scales_with_board_size(self) -> None:
        board = board_with(3, {"a1": "wwww"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 4))
        assert problem is not None and "carry limit" in problem
        assert slide_problem(board, WHITE, slide("a1", ">", 2, 1)) is None

    def test_cannot_run_off_the_board(self) -> None:
        board = board_with(5, {"e1": "w", "b1": "ww"})
        problem = slide_problem(board, WHITE, slide("e1", ">", 1))
        assert problem is not None and "edge" in problem
        assert slide_problem(board, WHITE, slide("b1", "<", 1, 1)) is not None
        assert slide_problem(board, WHITE, slide("b1", "-", 1)) is not None

    def test_may_land_on_flats_of_either_color(self) -> None:
        board = board_with(5, {"a1": "w", "b1": "w", "a2": "b"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1)) is None
        assert slide_problem(board, WHITE, slide("a1", "+", 1)) is None

    def test_may_pass_over_flats_dropping_one_each(self) -> None:
        board = board_with(5, {"a1": "www", "b1": "b", "c1": "w"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1, 1, 1)) is None

    def test_cannot_move_onto_a_wall(self) -> None:
        board = board_with(5, {"a1": "w", "b1": "Sb"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 1))
        assert problem is not None and "wall" in problem

    def test_cannot_move_onto_a_capstone(self) -> None:
        board = board_with(5, {"a1": "w", "b1": "Cb"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 1))
        assert problem is not None and "capstone" in problem

    def test_cannot_pass_over_a_wall(self) -> None:
        board = board_with(5, {"a1": "ww", "b1": "Sb"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 1, 1))
        assert problem is not None and "wall" in problem

    def test_cannot_pass_over_a_capstone(self) -> None:
        board = board_with(5, {"a1": "ww", "b1": "Cb"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1, 1)) is not None

    def test_a_wall_can_move_onto_flats_but_not_onto_walls(self) -> None:
        board = board_with(5, {"a1": "Sw", "b1": "b", "a2": "Sb"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1)) is None
        assert slide_problem(board, WHITE, slide("a1", "+", 1)) is not None

    def test_a_capstone_flattens_a_wall(self) -> None:
        board = board_with(5, {"a1": "Cw", "b1": "Sb"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1)) is None

    def test_a_capstone_flattens_a_wall_of_its_own_color(self) -> None:
        board = board_with(5, {"a1": "Cw", "b1": "Sw"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1)) is None

    def test_a_capstone_flattens_a_wall_on_top_of_a_stack(self) -> None:
        board = board_with(5, {"a1": "Cw", "b1": "bwSb"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1)) is None

    def test_a_capstone_in_a_stack_flattens_if_it_arrives_alone(self) -> None:
        board = board_with(5, {"a1": "wCw", "c1": "Sb"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1, 1)) is None

    def test_a_capstone_cannot_flatten_a_wall_with_other_stones(self) -> None:
        board = board_with(5, {"a1": "wCw", "b1": "Sb"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 2))
        assert problem is not None and "alone" in problem

    def test_a_capstone_cannot_pass_over_a_wall(self) -> None:
        board = board_with(5, {"a1": "wCw", "b1": "Sb"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 1, 1))
        assert problem is not None and "wall" in problem

    def test_a_capstone_cannot_move_onto_a_capstone(self) -> None:
        board = board_with(5, {"a1": "Cw", "b1": "Cb"})
        assert slide_problem(board, WHITE, slide("a1", ">", 1)) is not None

    def test_a_flat_cannot_flatten_a_wall(self) -> None:
        board = board_with(5, {"a1": "w", "b1": "Sb"})
        problem = slide_problem(board, WHITE, slide("a1", ">", 1))
        assert problem is not None and "capstone" in problem


class TestDropProblem:
    square = Square(1, 1)

    def test_empty_squares_and_flats_accept_drops(self) -> None:
        dropped = stack_of("w")
        assert drop_problem(self.square, None, dropped, True) is None
        assert drop_problem(self.square, stack_of("b")[-1], dropped, False) is None

    def test_walls_accept_only_a_lone_final_capstone(self) -> None:
        wall = stack_of("Sb")[-1]
        cap = stack_of("Cw")
        assert drop_problem(self.square, wall, cap, True) is None
        assert drop_problem(self.square, wall, cap, False) is not None
        assert drop_problem(self.square, wall, stack_of("wCw"), True) is not None
        assert drop_problem(self.square, wall, stack_of("w"), True) is not None

    def test_capstones_accept_nothing(self) -> None:
        cap = stack_of("Cb")[-1]
        assert drop_problem(self.square, cap, stack_of("Cw"), True) is not None
