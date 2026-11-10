"""Move generation for either player, which threat detection is built on.

Spotting a threat means asking what the player who just moved could do if
they moved again, so moves must be listable for a player whose turn it isn't.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from tak.engine import Color, Game, PieceType, PlaceMove, SlideMove, Square
from tak.engine.moves import regular_moves

Started = Callable[..., Game]
Position = Callable[..., Game]
RandomPositions = Callable[[int, int, int], Iterator[Game]]


@pytest.mark.parametrize("size", [3, 4, 5, 6])
@pytest.mark.parametrize("seed", range(3))
def test_the_player_to_move_gets_exactly_their_legal_moves(
    size: int, seed: int, random_positions: RandomPositions
) -> None:
    for game in random_positions(size, seed, 120):
        if game.is_over or game.in_opening:
            continue
        mover = game.to_move
        assert regular_moves(game.board, mover, game.reserve(mover)) == game.legal_moves()


def test_moves_for_the_player_who_is_not_to_move(started: Started) -> None:
    game = started("a1 e5 c3", size=5)  # White owns e5 and c3; Black is to move
    assert game.to_move is Color.BLACK
    moves = regular_moves(game.board, Color.WHITE, game.reserve(Color.WHITE))
    slides = {move.start for move in moves if isinstance(move, SlideMove)}
    assert slides == {Square.from_name("e5"), Square.from_name("c3")}
    assert PlaceMove(Square.from_name("b2"), PieceType.CAP) in moves


def test_placements_follow_the_given_reserve(position: Position) -> None:
    game = position(5, {"a1": "w"}, white=(0, 1), black=(3, 0))
    white = regular_moves(game.board, Color.WHITE, game.reserve(Color.WHITE))
    black = regular_moves(game.board, Color.BLACK, game.reserve(Color.BLACK))
    assert {m.piece for m in white if isinstance(m, PlaceMove)} == {PieceType.CAP}
    assert {m.piece for m in black if isinstance(m, PlaceMove)} == {PieceType.FLAT, PieceType.WALL}
    assert not any(isinstance(m, SlideMove) for m in black)  # Black controls no stacks
