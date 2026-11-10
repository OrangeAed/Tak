"""Regression test for the road search, which threat detection runs very often.

To make threat detection fast, ``find_road`` now starts its search from the
starting edge directly instead of scanning every square for it. This checks
that it still finds exactly the same road as the original full scan.
"""

from __future__ import annotations

import random
from collections import deque

import pytest

from tak.engine import BOARD_CONFIG, Board, Color, Direction, Piece, PieceType, Square, find_road


def reference_road(board: Board, color: Color) -> list[Square] | None:
    """The original search: breadth first from every edge square found by a full scan."""

    def is_road(square: Square) -> bool:
        top = board.top(square)
        return top is not None and top.color is color and top.counts_for_roads

    for axis in (0, 1):
        parent: dict[Square, Square | None] = {}
        queue: deque[Square] = deque()
        for square in board.squares():
            if square[axis] == 0 and is_road(square):
                parent[square] = None
                queue.append(square)
        while queue:
            square = queue.popleft()
            if square[axis] == board.size - 1:
                path: list[Square] = []
                node: Square | None = square
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]
            for direction in Direction:
                neighbor = square.step(direction)
                if board.in_bounds(neighbor) and neighbor not in parent and is_road(neighbor):
                    parent[neighbor] = square
                    queue.append(neighbor)
    return None


def random_board(size: int, rng: random.Random) -> Board:
    """Mostly flats, mostly of one color, so roads (and near misses) are common."""
    board = Board(size)
    kinds = [PieceType.FLAT] * 6 + [PieceType.WALL, PieceType.CAP]
    main = rng.choice(list(Color))
    for square in board.squares():
        if rng.random() < 0.8:
            color = main if rng.random() < 0.7 else main.opponent
            board.set_stack(square, (Piece(color, rng.choice(kinds)),))
    return board


@pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
def test_find_road_matches_the_original_search(size: int) -> None:
    rng = random.Random(size)
    roads = 0
    for _ in range(300):
        board = random_board(size, rng)
        for color in Color:
            road = find_road(board, color)
            assert road == reference_road(board, color)
            roads += road is not None
    assert roads >= 50  # the boards really did exercise roads
