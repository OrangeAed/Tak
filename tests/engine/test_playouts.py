"""Randomized whole-game tests.

These play many random legal games and check properties that must hold in
every position, which catches bugs that hand-written scenarios miss.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from tak.engine import (
    BOARD_CONFIG,
    Color,
    Direction,
    Game,
    IllegalMoveError,
    Move,
    PieceType,
    PlaceMove,
    Reason,
    SlideMove,
    find_road,
    game_from_ptn,
    game_to_ptn,
)

Snapshot = Callable[[Game], Any]


def all_candidate_moves(game: Game) -> Iterator[Move]:
    """Every syntactically valid move, legal or not, for cross-checking the generator."""
    size = game.size

    def compositions(total: int, max_parts: int) -> Iterator[tuple[int, ...]]:
        if total == 0:
            yield ()
        elif max_parts > 0:
            for first in range(1, total + 1):
                for rest in compositions(total - first, max_parts - 1):
                    yield (first, *rest)

    for square in game.board.squares():
        for piece in PieceType:
            yield PlaceMove(square, piece)
        for direction in Direction:
            # One more than the carry limit, so over-limit moves are tried too.
            for count in range(1, size + 2):
                for drops in compositions(count, size):
                    yield SlideMove(square, direction, drops)


def is_legal(game: Game, move: Move) -> bool:
    try:
        game.check(move)
    except IllegalMoveError:
        return False
    return True


def pieces_of(game: Game, color: Color) -> int:
    return sum(
        1
        for square in game.board.squares()
        for piece in game.board.stack(square)
        if piece.color is color
    )


def assert_result_is_consistent(game: Game) -> None:
    white_road = find_road(game.board, Color.WHITE) is not None
    black_road = find_road(game.board, Color.BLACK) is not None
    out_of_pieces = any(reserve.total == 0 for reserve in game.reserves.values())
    result = game.result

    if result is None:
        assert not white_road and not black_road
        assert not out_of_pieces and not game.board.is_full()
    elif result.reason is Reason.ROAD:
        assert result.winner is not None
        assert find_road(game.board, result.winner) is not None
        if white_road and black_road:
            just_moved = Color.WHITE if (game.ply - 1) % 2 == 0 else Color.BLACK
            assert result.winner is just_moved  # double road: the mover wins
    else:
        assert not white_road and not black_road
        assert out_of_pieces or game.board.is_full()


def random_game(size: int, seed: int, max_plies: int) -> Iterator[Game]:
    """Play a random game, yielding the game after every move."""
    rng = random.Random(seed * 1000 + size)
    game = Game(size)
    while not game.is_over and game.ply < max_plies:
        moves = game.legal_moves()
        assert moves, "a game in progress must always have a legal move"
        game.play(rng.choice(moves))
        yield game


@pytest.mark.parametrize("size", [3, 4, 5, 6])
@pytest.mark.parametrize("seed", range(6))
def test_random_games_keep_every_invariant(size: int, seed: int, snapshot: Snapshot) -> None:
    supply = sum(BOARD_CONFIG[size])
    for game in random_game(size, seed, max_plies=250):
        # No piece is ever created or destroyed.
        for color in Color:
            assert pieces_of(game, color) + game.reserve(color).total == supply
        assert_result_is_consistent(game)

        # Taking a move back and replaying it gives exactly the same position.
        after = snapshot(game)
        move = game.history[-1].move
        game.undo()
        game.play(move)
        assert snapshot(game) == after


def final_position(size: int, seed: int, max_plies: int) -> Game:
    """The last position of a random game."""
    last = deque(random_game(size, seed, max_plies), maxlen=1)
    assert last, "the game should have at least one move"
    return last[0]


@pytest.mark.parametrize("seed", range(10))
def test_small_random_games_always_finish(seed: int) -> None:
    assert final_position(3, seed, max_plies=400).is_over


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("seed", range(3))
def test_generated_moves_are_exactly_the_legal_moves(size: int, seed: int) -> None:
    for game in random_game(size, seed, max_plies=80):
        if game.ply % 3 != 0:
            continue
        generated = game.legal_moves()
        assert len(generated) == len(set(generated)), "duplicate moves generated"
        expected = {move for move in all_candidate_moves(game) if is_legal(game, move)}
        assert set(generated) == expected


@pytest.mark.parametrize("size", [3, 4, 5, 6])
@pytest.mark.parametrize("seed", range(4))
def test_random_games_survive_a_ptn_round_trip(size: int, seed: int, snapshot: Snapshot) -> None:
    game = final_position(size, seed, max_plies=150)
    replayed = game_from_ptn(game_to_ptn(game))
    assert snapshot(replayed) == snapshot(game)
    assert [p.move for p in replayed.history] == [p.move for p in game.history]
    assert [p.flattened for p in replayed.history] == [p.flattened for p in game.history]
