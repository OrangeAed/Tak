"""Tests for move objects and legal move generation."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tak.engine import (
    Direction,
    Game,
    PieceType,
    PlaceMove,
    SlideMove,
    Square,
    legal_moves,
    parse_move,
)

Started = Callable[..., Game]


def sq(name: str) -> Square:
    return Square.from_name(name)


class TestMoveObjects:
    def test_placing_defaults_to_a_flat_stone(self) -> None:
        assert PlaceMove(sq("a1")).piece is PieceType.FLAT

    def test_slide_count_is_the_sum_of_drops(self) -> None:
        assert SlideMove(sq("a1"), Direction.RIGHT, (2, 1, 1)).count == 4

    @pytest.mark.parametrize("drops", [(), (0,), (1, 0), (2, -1)])
    def test_slides_must_drop_a_stone_on_every_square(self, drops: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="at least one stone"):
            SlideMove(sq("a1"), Direction.RIGHT, drops)

    def test_moves_are_hashable_and_comparable(self) -> None:
        assert parse_move("3c2>12") == SlideMove(sq("c2"), Direction.RIGHT, (1, 2))
        assert len({parse_move("a1"), parse_move("Fa1"), parse_move("Sa1")}) == 2


class TestOpeningMoves:
    def test_only_flat_placements_on_every_empty_square(self) -> None:
        moves = Game(5).legal_moves()
        assert len(moves) == 25
        assert all(isinstance(m, PlaceMove) and m.piece is PieceType.FLAT for m in moves)

    def test_second_opening_move_excludes_the_taken_square(self, started: Started) -> None:
        moves = started("a1").legal_moves()
        assert len(moves) == 24
        assert PlaceMove(sq("a1")) not in moves


class TestRegularMoves:
    def test_five_by_five_after_the_opening(self, started: Started) -> None:
        # 23 empty squares x (flat, wall, capstone), plus White's lone corner stone
        # on e5 which can step left or down.
        moves = started("a1 e5").legal_moves()
        assert len(moves) == 23 * 3 + 2

    def test_small_boards_offer_no_capstones(self, started: Started) -> None:
        moves = started("a1 c3", size=3).legal_moves()
        placements = [m for m in moves if isinstance(m, PlaceMove)]
        assert len(placements) == 7 * 2
        assert all(m.piece is not PieceType.CAP for m in placements)
        assert len(moves) == 14 + 2  # c3 can step left or down

    def test_no_capstone_placements_once_it_is_used(self, started: Started) -> None:
        moves = started("a1 e5 Cc3 b5").legal_moves()
        assert not any(isinstance(m, PlaceMove) and m.piece is PieceType.CAP for m in moves)

    def test_only_your_own_stacks_can_move(self, started: Started) -> None:
        game = started("a1 e5")  # a1 is black, e5 is white; White to move
        starts = {m.start for m in game.legal_moves() if isinstance(m, SlideMove)}
        assert starts == {sq("e5")}

    def test_a_capstone_may_flatten_but_a_flat_may_not(self, started: Started) -> None:
        # White cap on c3, Black wall on d3, White flat on b3; Black plays a filler.
        game = started("a1 e5 Cc3 Sd3 b3 a5")
        moves = set(game.legal_moves())
        assert SlideMove(sq("c3"), Direction.RIGHT, (1,)) in moves  # cap flattens the wall
        assert SlideMove(sq("b3"), Direction.RIGHT, (1,)) not in moves  # onto the cap: blocked

    def test_walls_stop_a_flat_stack(self, started: Started) -> None:
        game = started("a1 e5 c3 Sd3")  # White flat c3 beside a Black wall
        moves = set(game.legal_moves())
        assert SlideMove(sq("c3"), Direction.RIGHT, (1,)) not in moves
        assert SlideMove(sq("c3"), Direction.LEFT, (1,)) in moves

    def test_every_generated_move_can_be_played(self, started: Started) -> None:
        game = started("a1 e5 c2 c3 c2+ a5 c2 e1 c2+ b5")
        moves = game.legal_moves()
        assert moves
        for move in moves:
            game.check(move)  # raises if the generator produced an illegal move

    def test_tall_stacks_offer_every_split(self, started: Started) -> None:
        # c3 holds three stones controlled by White; moving right over empty d3/e3
        # allows these drop patterns for each pickup size.
        game = started("a1 e5 c2 c3 c2+ a5 c2 e1 c2+ b5")
        slides = {m for m in game.legal_moves() if isinstance(m, SlideMove)}
        right = {m.drops for m in slides if m.start == sq("c3") and m.direction is Direction.RIGHT}
        assert right == {(1,), (2,), (3,), (1, 1), (1, 2), (2, 1)}

    def test_no_moves_once_the_game_is_over(self, started: Started) -> None:
        game = started("a1 a2 b2 a3 c2", size=3)
        assert game.legal_moves() == []
        assert legal_moves(game) == []
