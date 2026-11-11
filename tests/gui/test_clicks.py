"""Click to move: turning clicks on squares into moves."""

from __future__ import annotations

import random

import pytest

from tak.engine import (
    Color,
    Direction,
    Game,
    Piece,
    PieceType,
    PlaceMove,
    SlideMove,
    Square,
    parse_move,
)
from tak.gui.clicks import Lift, MoveBuilder, Outcome, touched_squares

W, B = Piece(Color.WHITE, PieceType.FLAT), Piece(Color.BLACK, PieceType.FLAT)
W_CAP, B_WALL = Piece(Color.WHITE, PieceType.CAP), Piece(Color.BLACK, PieceType.WALL)


def sq(name: str) -> Square:
    return Square.from_name(name)


def position(size: int = 5, **stacks: tuple[Piece, ...]) -> Game:
    """A game past the opening with White to move, and the given stacks set up."""
    game = Game(size)
    game.play(parse_move("a1"))
    game.play(parse_move(f"{chr(ord('a') + size - 1)}{size}"))
    for name, stack in stacks.items():
        game.board.set_stack(sq(name), stack)
    return game


def clicks(builder: MoveBuilder, game: Game, *names: str) -> Outcome:
    outcome = Outcome()
    for name in names:
        outcome = builder.click(game, sq(name))
    return outcome


class TestPlacing:
    def test_click_an_empty_square_to_place_a_flat(self) -> None:
        outcome = MoveBuilder().click(position(), sq("c3"))
        assert outcome.move == PlaceMove(sq("c3"), PieceType.FLAT)

    @pytest.mark.parametrize("piece", [PieceType.WALL, PieceType.CAP])
    def test_walls_and_capstones_when_chosen(self, piece: PieceType) -> None:
        outcome = MoveBuilder().click(position(), sq("c3"), piece)
        assert outcome.move == PlaceMove(sq("c3"), piece)

    def test_the_opening_always_places_a_flat(self) -> None:
        outcome = MoveBuilder().click(Game(5), sq("c3"), PieceType.WALL)
        assert outcome.move == PlaceMove(sq("c3"), PieceType.FLAT)

    def test_no_capstones_on_small_boards(self) -> None:
        outcome = MoveBuilder().click(position(3), sq("b2"), PieceType.CAP)
        assert outcome.move is None and outcome.error
        assert "no capstones" in (outcome.message or "")

    def test_nothing_happens_once_the_game_is_over(self) -> None:
        game = Game(3)
        for move in ["a1", "c3", "b1", "b3", "b2", "a3", "c2"]:
            game.play(parse_move(move))
        outcome = MoveBuilder().click(game, sq("a2"))
        assert outcome.move is None and outcome.error


class TestLifting:
    def test_lifts_as_many_as_the_carry_limit_allows(self) -> None:
        builder = MoveBuilder()
        builder.click(position(c3=(B, W, B, W, B, W, W)), sq("c3"))
        assert builder.lift == Lift(sq("c3"), 5)  # 7 stones, but the limit on 5x5 is 5

    def test_clicking_the_stack_again_carries_fewer_then_puts_it_back(self) -> None:
        builder, game = MoveBuilder(), position(c3=(B, W))
        clicks(builder, game, "c3", "c3")
        assert builder.lift == Lift(sq("c3"), 1)
        outcome = clicks(builder, game, "c3")
        assert builder.lift is None and outcome.move is None
        assert "back" in (outcome.message or "")

    def test_only_your_own_stacks(self) -> None:
        builder = MoveBuilder()
        outcome = builder.click(position(c3=(W, B)), sq("c3"))
        assert builder.lift is None and outcome.error

    def test_no_stack_moves_in_the_opening(self) -> None:
        game = Game(5)
        game.play(parse_move("a1"))  # White places Black's first flat
        outcome = MoveBuilder().click(game, sq("a1"))  # Black's turn, still the opening
        assert outcome.error and "first turn" in (outcome.message or "")

    def test_switch_to_another_stack_before_dropping(self) -> None:
        builder, game = MoveBuilder(), position(c3=(W,), b2=(B, W))
        clicks(builder, game, "c3", "b2")
        assert builder.lift == Lift(sq("b2"), 2)

    def test_cancel(self) -> None:
        builder = MoveBuilder()
        builder.click(position(c3=(W,)), sq("c3"))
        assert builder.cancel() is True
        assert builder.lift is None
        assert builder.cancel() is False


class TestMovingStacks:
    def test_one_stone_one_square(self) -> None:
        outcome = clicks(MoveBuilder(), position(c3=(W,)), "c3", "c4")
        assert outcome.move == parse_move("c3+")

    def test_spreading_a_stack(self) -> None:
        builder, game = MoveBuilder(), position(c3=(B, W, W))
        outcome = clicks(builder, game, "c3", "c4", "c4")
        assert outcome.move is None
        assert builder.lift == Lift(sq("c3"), 3, Direction.UP, (2,))
        outcome = clicks(builder, game, "c5")
        assert outcome.move == parse_move("3c3+21")
        assert builder.lift is None

    def test_carrying_fewer_then_moving(self) -> None:
        game = position(c3=(B, W, B, W, W))
        outcome = clicks(MoveBuilder(), game, "c3", "c3", "c3", "c3", "b3", "a3")
        assert outcome.move == parse_move("2c3<11")

    def test_cannot_drop_a_flat_on_a_wall(self) -> None:
        builder, game = MoveBuilder(), position(c3=(W,), c4=(B_WALL,))
        builder.click(game, sq("c3"))
        outcome = builder.click(game, sq("c4"))
        assert outcome.move is None and outcome.error
        assert builder.lift == Lift(sq("c3"), 1)  # still holding it

    def test_a_capstone_flattens_a_wall(self) -> None:
        game = position(c3=(W_CAP,), c4=(B_WALL,))
        outcome = clicks(MoveBuilder(), game, "c3", "c4")
        assert outcome.move == parse_move("c3+")
        game.play(outcome.move)
        assert game.history[-1].flattened

    def test_clicks_off_the_path_are_explained(self) -> None:
        builder, game = MoveBuilder(), position(c3=(B, W, W))
        clicks(builder, game, "c3", "c4")
        lift = builder.lift
        outcome = builder.click(game, sq("e1"))
        assert outcome.error and "c4" in (outcome.message or "")
        assert builder.lift == lift

    def test_describe_says_what_to_do(self) -> None:
        builder, game = MoveBuilder(), position(c3=(B, W, W))
        assert builder.describe() is None
        builder.click(game, sq("c3"))
        assert "Carrying 3 stones from c3" in (builder.describe() or "")
        builder.click(game, sq("c4"))
        assert "Carrying 2 more stones" in (builder.describe() or "")


class TestTargetsAndPreview:
    def test_targets_are_the_useful_squares(self) -> None:
        builder, game = MoveBuilder(), position(c3=(W,), c4=(B_WALL,))
        builder.click(game, sq("c3"))
        # c3 itself (to put it back) and every neighbour except the wall.
        assert builder.targets(game) == {sq("c3"), sq("b3"), sq("d3"), sq("c2")}

    def test_targets_after_a_drop(self) -> None:
        builder, game = MoveBuilder(), position(c3=(B, W, W))
        clicks(builder, game, "c3", "c4")
        assert builder.targets(game) == {sq("c4"), sq("c5")}

    def test_no_targets_without_a_lift(self) -> None:
        assert MoveBuilder().targets(position()) == set()

    def test_preview_shows_the_move_so_far(self) -> None:
        builder, game = MoveBuilder(), position(c3=(B, W, B, W))
        clicks(builder, game, "c3", "c3", "c4")  # carry 3: W B W, drop the W on c4
        board, floating = builder.preview(game.board)
        assert board.stack(sq("c3")) == (B,)
        assert board.stack(sq("c4")) == (W,)
        assert floating == (sq("c4"), (B, W))
        assert game.board.stack(sq("c3")) == (B, W, B, W)  # the real board is untouched

    def test_no_preview_without_a_lift(self) -> None:
        game = position()
        assert MoveBuilder().preview(game.board) == (game.board, None)


def test_touched_squares() -> None:
    assert touched_squares(parse_move("Sc3")) == [sq("c3")]
    assert touched_squares(parse_move("3c3>12")) == [sq("c3"), sq("d3"), sq("e3")]


def click_sequence(game: Game, move: SlideMove) -> list[Square]:
    """The clicks a player would make for ``move``: lift, put some back, drop along."""
    lifted = max(
        m.count for m in game.legal_moves() if isinstance(m, SlideMove) and m.start == move.start
    )
    sequence = [move.start] * (1 + lifted - move.count)
    square = move.start
    for drop in move.drops:
        square = square.step(move.direction)
        sequence += [square] * drop
    return sequence


@pytest.mark.parametrize("size,seed", [(4, 1), (5, 2), (5, 3), (6, 4), (7, 5)])
def test_every_legal_move_can_be_clicked(size: int, seed: int) -> None:
    """In real positions from random games, each legal move comes out of its clicks exactly."""
    rng = random.Random(seed)
    game = Game(size)
    checked = 0
    while not game.is_over and game.ply < 60:
        if game.ply % 6 == 5:
            moves = game.legal_moves()
            for move in rng.sample(moves, min(25, len(moves))):
                builder = MoveBuilder()
                if isinstance(move, PlaceMove):
                    outcome = builder.click(game, move.square, move.piece)
                else:
                    for square in click_sequence(game, move):
                        outcome = builder.click(game, square)
                assert outcome.move == move, (move, outcome)
                checked += 1
        game.play(rng.choice(game.legal_moves()))
    assert checked > 50
