"""Tests for whole-game behavior: turn order, openings, wins, and undo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tak.engine import (
    Color,
    Game,
    GameOverError,
    GameResult,
    IllegalMoveError,
    Piece,
    PieceType,
    PlaceMove,
    Reason,
    Reserve,
    Square,
    count_flats,
    find_road,
    format_move,
    game_to_ptn,
    parse_move,
)

Started = Callable[..., Game]
Snapshot = Callable[[Game], Any]

WHITE, BLACK = Color.WHITE, Color.BLACK
WHITE_FLAT = Piece(WHITE, PieceType.FLAT)
BLACK_FLAT = Piece(BLACK, PieceType.FLAT)
BLACK_WALL = Piece(BLACK, PieceType.WALL)
WHITE_CAP = Piece(WHITE, PieceType.CAP)


def sq(name: str) -> Square:
    return Square.from_name(name)


class TestNewGame:
    def test_initial_state(self) -> None:
        game = Game(5)
        assert game.size == 5
        assert game.to_move is WHITE
        assert game.in_opening
        assert game.ply == 0
        assert game.turn_number == 1
        assert game.result is None
        assert not game.is_over
        assert game.reserve(WHITE) == Reserve(21, 1)
        assert game.reserve(BLACK) == Reserve(21, 1)

    @pytest.mark.parametrize(
        ("size", "stones", "caps"),
        [(3, 10, 0), (4, 15, 0), (5, 21, 1), (6, 30, 1), (7, 40, 2), (8, 50, 2)],
    )
    def test_reserves_follow_the_rules_table(self, size: int, stones: int, caps: int) -> None:
        game = Game(size)
        for color in Color:
            assert game.reserve(color) == Reserve(stones, caps)

    def test_default_size_is_five(self) -> None:
        assert Game().size == 5

    def test_unsupported_size(self) -> None:
        with pytest.raises(ValueError):
            Game(9)


class TestOpening:
    def test_each_player_places_a_flat_of_the_opponents_color(self, started: Started) -> None:
        game = started("a1")
        assert game.board.stack(sq("a1")) == (BLACK_FLAT,)  # White placed a black stone
        assert game.reserve(BLACK) == Reserve(20, 1)
        assert game.reserve(WHITE) == Reserve(21, 1)

        game.play(parse_move("e5"))
        assert game.board.stack(sq("e5")) == (WHITE_FLAT,)  # Black placed a white stone
        assert game.reserve(WHITE) == Reserve(20, 1)

    def test_regular_play_starts_with_white(self, started: Started) -> None:
        game = started("a1 e5")
        assert not game.in_opening
        assert game.to_move is WHITE
        assert game.turn_number == 2

    def test_turns_alternate(self, started: Started) -> None:
        game = Game(5)
        movers = []
        for token in ["a1", "e5", "c3", "b2", "d4"]:
            movers.append(game.to_move)
            game.play(parse_move(token))
        assert movers == [WHITE, BLACK, WHITE, BLACK, WHITE]
        assert game.to_move is BLACK

    @pytest.mark.parametrize("token", ["Sa1", "Ca1"])
    def test_no_walls_or_capstones_on_the_first_turn(
        self, started: Started, token: str, snapshot: Snapshot
    ) -> None:
        game = started()
        before = snapshot(game)
        with pytest.raises(IllegalMoveError, match="first turn"):
            game.play(parse_move(token))
        assert snapshot(game) == before

    def test_black_may_not_open_with_a_wall_either(self, started: Started) -> None:
        game = started("a1")
        with pytest.raises(IllegalMoveError, match="first turn"):
            game.play(parse_move("Sb2"))

    def test_no_moving_stacks_on_the_first_turn(self, started: Started) -> None:
        game = started("a1")
        with pytest.raises(IllegalMoveError, match="first turn"):
            game.play(parse_move("a1>"))


class TestPlacing:
    def test_cannot_place_on_an_occupied_square(self, started: Started) -> None:
        game = started("a1 e5")
        with pytest.raises(IllegalMoveError, match="occupied"):
            game.play(parse_move("a1"))
        with pytest.raises(IllegalMoveError, match="occupied"):
            game.play(parse_move("e5"))
        assert game.ply == 2  # nothing was played

    def test_placing_uses_your_own_pieces(self, started: Started) -> None:
        game = started("a1 e5 c3")
        assert game.board.stack(sq("c3")) == (WHITE_FLAT,)
        assert game.reserve(WHITE) == Reserve(19, 1)
        assert game.reserve(BLACK) == Reserve(20, 1)

    def test_walls_use_stones_and_capstones_use_caps(self, started: Started) -> None:
        game = started("a1 e5 Sc3 Cd3")
        assert game.board.stack(sq("c3")) == (Piece(WHITE, PieceType.WALL),)
        assert game.board.stack(sq("d3")) == (Piece(BLACK, PieceType.CAP),)
        assert game.reserve(WHITE) == Reserve(19, 1)
        assert game.reserve(BLACK) == Reserve(20, 0)

    def test_only_one_capstone_on_a_five_by_five_board(self, started: Started) -> None:
        game = started("a1 e5 Cc3 b5")
        with pytest.raises(IllegalMoveError, match="no capstones left"):
            game.play(parse_move("Cd3"))

    def test_no_capstones_on_small_boards(self, started: Started) -> None:
        game = started("a1 c3", size=3)
        with pytest.raises(IllegalMoveError, match="no capstones"):
            game.play(PlaceMove(sq("b2"), PieceType.CAP))

    def test_off_board_squares_are_rejected(self, started: Started) -> None:
        game = started("a1 c3", size=3)
        with pytest.raises(IllegalMoveError, match="not on the board"):
            game.play(PlaceMove(Square(3, 0)))

    def test_running_out_of_stones_blocks_placing(self, started: Started) -> None:
        game = started("a1 c3", size=3)
        game.reserves[WHITE] = Reserve(0, 0)
        with pytest.raises(IllegalMoveError, match="no stones left"):
            game.play(parse_move("b2"))


class TestSliding:
    def test_cannot_move_the_opponents_stack(self, started: Started) -> None:
        game = started("a1 e5")
        with pytest.raises(IllegalMoveError, match="don't control"):
            game.play(parse_move("a1+"))  # a1 is a black stone and White is to move

    def test_cannot_move_from_an_empty_square(self, started: Started) -> None:
        game = started("a1 e5")
        with pytest.raises(IllegalMoveError, match="no stones"):
            game.play(parse_move("c3>"))

    def test_a_single_stone_moves_and_the_origin_empties(self, started: Started) -> None:
        game = started("a1 e5 c3 b5 c3>")
        assert game.board.stack(sq("c3")) == ()
        assert game.board.stack(sq("d3")) == (WHITE_FLAT,)

    def test_moving_onto_a_stack_covers_it(self, started: Started) -> None:
        game = started("a1 e5 c3 c4 c3+")
        assert game.board.stack(sq("c4")) == (BLACK_FLAT, WHITE_FLAT)
        assert game.board.top(sq("c4")) == WHITE_FLAT

    def test_a_stack_drops_from_the_bottom_and_keeps_its_order(self, started: Started) -> None:
        # Build c3 = [black, white, white], then carry all three: 1 on d3, 2 on e3.
        game = started("a1 e5 c2 c3 c2+ a5 c2 e1 c2+ b5")
        assert game.board.stack(sq("c3")) == (BLACK_FLAT, WHITE_FLAT, WHITE_FLAT)
        game.play(parse_move("3c3>12"))
        assert game.board.stack(sq("c3")) == ()
        assert game.board.stack(sq("d3")) == (BLACK_FLAT,)
        assert game.board.stack(sq("e3")) == (WHITE_FLAT, WHITE_FLAT)

    def test_picking_up_part_of_a_stack_leaves_the_rest(self, started: Started) -> None:
        game = started("a1 e5 c2 c3 c2+ a5 c2 e1 c2+ b5")
        game.play(parse_move("2c3+"))
        assert game.board.stack(sq("c3")) == (BLACK_FLAT,)
        assert game.board.stack(sq("c4")) == (WHITE_FLAT, WHITE_FLAT)

    def test_walls_block_slides(self, started: Started) -> None:
        game = started("a1 e5 c3 Sd3")
        with pytest.raises(IllegalMoveError, match="wall"):
            game.play(parse_move("c3>"))

    def test_a_capstone_flattens_a_wall(self, started: Started) -> None:
        game = started("a1 e5 Cc3 Sd3")
        game.play(parse_move("c3>"))
        assert game.board.stack(sq("c3")) == ()
        assert game.board.stack(sq("d3")) == (BLACK_FLAT, WHITE_CAP)
        assert game.history[-1].flattened

    def test_flattening_a_wall_on_a_stack_keeps_the_stack_beneath(self, started: Started) -> None:
        game = started("a1 e5 Cc3 d3 b5 Sd4 b4 d4-")
        assert game.board.stack(sq("d3")) == (BLACK_FLAT, BLACK_WALL)
        game.play(parse_move("c3>"))
        assert game.board.stack(sq("d3")) == (BLACK_FLAT, BLACK_FLAT, WHITE_CAP)
        assert format_move(game.history[-1].move, flattened=game.history[-1].flattened) == "c3>*"

    def test_ordinary_moves_are_not_marked_as_flattening(self, started: Started) -> None:
        game = started("a1 e5 c3 b5 c3>")
        assert not game.history[-1].flattened

    def test_the_carry_limit_is_enforced(self, started: Started) -> None:
        # On 3x3 the carry limit is 3. White stacks four flats on b2 while Black
        # makes harmless corner and wall moves, then tries to lift all four.
        game = started("a1 a2 b2 c1 a2> a3 c2 c3 c2< Sb1 b3 Sa2 b3- Sc2", size=3)
        assert game.board.stack(sq("b2")) == (WHITE_FLAT,) * 4
        with pytest.raises(IllegalMoveError, match="carry limit"):
            game.play(parse_move("4b2+"))
        assert game.ply == 14  # the illegal move was not played

        game.play(parse_move("3b2+"))  # three is within the limit
        assert game.board.stack(sq("b2")) == (WHITE_FLAT,)
        assert game.board.stack(sq("b3")) == (WHITE_FLAT,) * 3


class TestWinning:
    def test_white_road(self, started: Started) -> None:
        game = started("a1 a2 b2 a3 c2", size=3)
        assert game.result == GameResult(WHITE, Reason.ROAD)
        assert game.result.ptn == "R-0"
        assert game.is_over

    def test_black_road(self, started: Started) -> None:
        game = started("a1 a2 c3 b1 a3 c1", size=3)
        assert game.result == GameResult(BLACK, Reason.ROAD)
        assert game.result.ptn == "0-R"

    def test_the_mover_wins_a_double_road(self, started: Started) -> None:
        # White uncovers Black's road and completes its own with the same move.
        game = started("a1 a3 c2 c1 c2- b1 b2 b3 a2 c3", size=3)
        assert game.result is None
        game.play(parse_move("c1+"))
        assert find_road(game.board, WHITE) is not None
        assert find_road(game.board, BLACK) is not None
        assert game.result == GameResult(WHITE, Reason.ROAD)

    def test_flat_win_when_the_board_fills(self, started: Started) -> None:
        game = started("a2 a3 c3 Sb1 Sb2 b3 a1 c2 c1", size=3)
        assert game.board.is_full()
        assert game.result == GameResult(WHITE, Reason.FLAT)
        assert (count_flats(game.board, WHITE), count_flats(game.board, BLACK)) == (4, 3)
        assert game.result.ptn == "F-0"

    def test_draw_when_the_board_fills_with_equal_flats(self, started: Started) -> None:
        game = started("a2 a3 c3 Sb1 Sb2 b3 Sa1 c2 c1", size=3)
        assert game.board.is_full()
        assert game.result == GameResult(None, Reason.FLAT)
        assert game.result.ptn == "1/2-1/2"

    def test_flat_win_when_a_player_places_their_last_piece(self, started: Started) -> None:
        game = started("a1 a3", size=3)
        game.reserves[WHITE] = Reserve(1, 0)
        game.play(parse_move("c1"))  # White's last piece
        assert game.reserve(WHITE).total == 0
        assert game.result == GameResult(WHITE, Reason.FLAT)  # 2 flats to 1

    def test_a_road_on_the_last_piece_beats_the_flat_count(self, started: Started) -> None:
        game = started("a1 a2 b2", size=3)  # White road still needs c2; Black to move
        game.play(parse_move("a3"))
        game.reserves[WHITE] = Reserve(1, 0)
        game.play(parse_move("c2"))
        assert game.result == GameResult(WHITE, Reason.ROAD)

    def test_no_moves_after_the_game_is_over(self, started: Started, snapshot: Snapshot) -> None:
        game = started("a1 a2 b2 a3 c2", size=3)
        before = snapshot(game)
        with pytest.raises(GameOverError):
            game.play(parse_move("c3"))
        assert snapshot(game) == before
        assert game.legal_moves() == []

    def test_game_over_error_is_an_illegal_move_error(self) -> None:
        assert issubclass(GameOverError, IllegalMoveError)


class TestUndo:
    def test_undoing_a_placement(self, started: Started, snapshot: Snapshot) -> None:
        game = started("a1 e5")
        before = snapshot(game)
        game.play(parse_move("Cc3"))
        assert game.undo() == parse_move("Cc3")
        assert snapshot(game) == before
        assert game.reserve(WHITE) == Reserve(20, 1)

    def test_undoing_an_opening_placement(self, started: Started, snapshot: Snapshot) -> None:
        game = started()
        before = snapshot(game)
        game.play(parse_move("a1"))
        game.undo()
        assert snapshot(game) == before
        assert game.in_opening and game.to_move is WHITE

    def test_undoing_a_slide(self, started: Started, snapshot: Snapshot) -> None:
        game = started("a1 e5 c2 c3 c2+ a5 c2 e1 c2+ b5")
        before = snapshot(game)
        game.play(parse_move("3c3>12"))
        game.undo()
        assert snapshot(game) == before

    def test_undoing_a_flatten_restores_the_wall(self, started: Started) -> None:
        game = started("a1 e5 Cc3 d3 b5 Sd4 b4 d4-")
        game.play(parse_move("c3>"))
        game.undo()
        assert game.board.stack(sq("d3")) == (BLACK_FLAT, BLACK_WALL)
        assert game.board.stack(sq("c3")) == (WHITE_CAP,)

    def test_undoing_the_winning_move_reopens_the_game(self, started: Started) -> None:
        game = started("a1 a2 b2 a3 c2", size=3)
        assert game.is_over
        game.undo()
        assert game.result is None
        game.play(parse_move("c3"))  # a different move is now possible
        assert game.ply == 5

    def test_undoing_everything_returns_to_the_start(
        self, started: Started, snapshot: Snapshot
    ) -> None:
        start = snapshot(Game(3))
        game = started("a1 a2 b2 a3 c2", size=3)
        for _ in range(5):
            game.undo()
        assert snapshot(game) == start

    def test_nothing_to_undo(self) -> None:
        with pytest.raises(ValueError, match="no moves to undo"):
            Game(5).undo()


def test_history_records_the_moves_played(started: Started) -> None:
    game = started("a1 e5 c3")
    assert [played.move for played in game.history] == [
        parse_move("a1"),
        parse_move("e5"),
        parse_move("c3"),
    ]


def test_ptn_output_for_a_short_game(started: Started) -> None:
    game = started("a1 a2 b2 a3 c2", size=3)
    assert game_to_ptn(game).endswith("3. c2 R-0\n")
