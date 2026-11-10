"""Tests for ``regular_moves``: the moves a given player could make on a regular turn.

Threat detection needs the moves of the player who has just moved, not only
of the player whose turn it is, so move generation takes the player as input.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator

import pytest

from tak.engine import (
    Color,
    Direction,
    Game,
    Move,
    PieceType,
    PlaceMove,
    Reserve,
    SlideMove,
    Square,
    legal_moves,
)
from tak.engine.moves import regular_moves
from tak.engine.rules import slide_problem

Position = Callable[..., Game]
Started = Callable[..., Game]


def random_positions(size: int, seed: int, max_plies: int) -> Iterator[Game]:
    rng = random.Random(seed * 31 + size)
    game = Game(size)
    while not game.is_over and game.ply < max_plies:
        game.play(rng.choice(game.legal_moves()))
        yield game


def compositions(total: int) -> Iterator[tuple[int, ...]]:
    """Every way to split ``total`` stones into drops of at least one."""
    if total == 0:
        yield ()
        return
    for first in range(1, total + 1):
        for rest in compositions(total - first):
            yield (first, *rest)


def reference_moves(game: Game, color: Color) -> set[Move]:
    """Regular-turn moves for ``color``, found by trying everything against the rules."""
    board = game.board
    reserve = game.reserve(color)
    moves: set[Move] = set()
    for square in board.squares():
        if not board.stack(square):
            for piece in PieceType:
                available = reserve.caps if piece is PieceType.CAP else reserve.stones
                if available > 0:
                    moves.add(PlaceMove(square, piece))
        for direction in Direction:
            for count in range(1, len(board.stack(square)) + 1):
                for drops in compositions(count):
                    slide = SlideMove(square, direction, drops)
                    if slide_problem(board, color, slide) is None:
                        moves.add(slide)
    return moves


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("seed", range(3))
def test_the_player_to_move_gets_exactly_their_legal_moves(size: int, seed: int) -> None:
    for game in random_positions(size, seed, max_plies=60):
        if game.is_over or game.in_opening:
            continue
        color = game.to_move
        assert regular_moves(game.board, color, game.reserve(color)) == legal_moves(game)


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("seed", range(3))
def test_the_player_who_just_moved_gets_their_moves_too(size: int, seed: int) -> None:
    for game in random_positions(size, seed, max_plies=60):
        if game.is_over or game.in_opening:
            continue
        color = game.to_move.opponent  # not their turn
        moves = regular_moves(game.board, color, game.reserve(color))
        assert len(moves) == len(set(moves))  # no duplicates
        assert set(moves) == reference_moves(game, color)


def test_only_your_own_stacks_can_move(started: Started) -> None:
    game = started("a1 e5 c3", size=5)  # White: e5, c3; Black: a1
    for color in Color:
        slides = [
            move
            for move in regular_moves(game.board, color, game.reserve(color))
            if isinstance(move, SlideMove)
        ]
        tops = [game.board.top(move.start) for move in slides]
        assert {top.color for top in tops if top is not None} == {color}
        assert None not in tops


class TestReserves:
    STACKS = {"a1": "w", "c3": "b"}

    def placements(self, game: Game, reserve: Reserve) -> set[PieceType]:
        return {
            move.piece
            for move in regular_moves(game.board, Color.WHITE, reserve)
            if isinstance(move, PlaceMove)
        }

    def test_stones_and_a_capstone(self, position: Position) -> None:
        game = position(5, self.STACKS)
        assert self.placements(game, Reserve(3, 1)) == set(PieceType)

    def test_no_capstones_left(self, position: Position) -> None:
        game = position(5, self.STACKS)
        assert self.placements(game, Reserve(3, 0)) == {PieceType.FLAT, PieceType.WALL}

    def test_only_a_capstone_left(self, position: Position) -> None:
        game = position(5, self.STACKS)
        assert self.placements(game, Reserve(0, 1)) == {PieceType.CAP}

    def test_nothing_left_means_only_slides(self, position: Position) -> None:
        game = position(5, self.STACKS)
        moves = regular_moves(game.board, Color.WHITE, Reserve(0, 0))
        assert moves
        assert all(isinstance(move, SlideMove) for move in moves)


def test_regular_turns_ignore_the_opening_rules() -> None:
    # On the very first turn only flats may be placed, but a regular turn allows walls.
    game = Game(3)
    moves = regular_moves(game.board, Color.WHITE, game.reserve(Color.WHITE))
    assert PlaceMove(Square.from_name("a1"), PieceType.WALL) in moves
    opening = legal_moves(game)
    assert all(isinstance(move, PlaceMove) and move.piece is PieceType.FLAT for move in opening)
