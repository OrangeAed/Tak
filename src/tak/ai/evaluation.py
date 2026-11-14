"""Measuring a position: how well each player is doing, as numbers.

These are the things the strategist tries to push in its favor when it has no
win, no block, and no Tinue to play. Each is measured for both players, and a
position's score is the weighted sum of the differences (ours minus theirs),
so making the opponent's numbers worse counts as much as improving our own.

- **Longest road:** how far across the board the player's best connected
  group of flats and capstones reaches, counted in squares along a rank or
  file. A road is a group that reaches all the way.
- **Road gap:** roughly how many more squares the player needs before a road
  is complete, counting empty squares as one and squares that would have to
  be captured as more. Fewer is better, so it counts against the player.
- **Center:** squares controlled near the middle, which connect in more
  directions and are harder to cut off, weighted more than squares at the edge.
- **Control:** squares with one of the player's pieces on top.
- **Flats:** flat stones on top, the count that decides a flat win. It matters
  more as the board fills and the reserves run down.
- **Captives:** the player's own stones held inside stacks they control,
  which can be spread out later.

All of this is ordinary arithmetic on the board, tuned by hand; nothing is learned.
"""

from __future__ import annotations

import functools
import heapq
from dataclasses import dataclass

from tak.ai.tactics import (
    BOTTOM,
    LEFT,
    RIGHT,
    TOP,
    Stacks,
    edges_of,
    groups,
    neighbors_of,
    road_cells,
    squares_of,
    stacks_of,
)
from tak.engine.board import BOARD_CONFIG, Board
from tak.engine.game import Game
from tak.engine.pieces import Color, PieceType


@dataclass(frozen=True)
class Features:
    """What :func:`features` measures for one player."""

    longest_road: int
    road_gap: int
    center: float
    control: int
    flats: int
    captives: int


@dataclass(frozen=True)
class Weights:
    """How much each feature difference is worth in a score."""

    longest_road: float = 4.0
    road_gap: float = 6.0
    center: float = 1.5
    control: float = 1.0
    flats: float = 1.5
    #: Extra weight for flats at the end of the game, scaled by how near it is.
    late_flats: float = 6.0
    captives: float = 0.3
    #: Bonus for a position where we threaten to win next move (Tak).
    threat: float = 8.0


DEFAULT_WEIGHTS = Weights()

# What a square costs when working out the road gap. Opponent capstones can't be
# covered or moved through, so they block the road outright.
_OWN = 0
_EMPTY = 1
_OWN_WALL = 2
_THEIRS = 2
_THEIR_WALL = 4


@functools.cache
def centrality(size: int) -> tuple[float, ...]:
    """For each square index, 1.0 in the middle of the board down to 0.0 in the corners."""
    middle = (size - 1) / 2
    if middle == 0:
        return (1.0,) * (size * size)
    return tuple(
        1.0 - (abs(square.file - middle) + abs(square.rank - middle)) / (2 * middle)
        for square in squares_of(size)
    )


def longest_road(stacks: Stacks, size: int, color: Color) -> int:
    """How many squares along a rank or file ``color``'s best connected group spans.

    A road spans the whole board, so this is ``size`` for a complete road.
    ``stacks`` is the board as :func:`tak.ai.tactics.stacks_of` gives it.
    """
    label, flags = groups(road_cells(stacks, color), size)
    if not flags:
        return 0
    low_file = [size] * len(flags)
    high_file = [-1] * len(flags)
    low_rank = [size] * len(flags)
    high_rank = [-1] * len(flags)
    for cell, square in enumerate(squares_of(size)):
        group = label[cell]
        if group < 0:
            continue
        low_file[group] = min(low_file[group], square.file)
        high_file[group] = max(high_file[group], square.file)
        low_rank[group] = min(low_rank[group], square.rank)
        high_rank[group] = max(high_rank[group], square.rank)
    return max(
        max(high_file[g] - low_file[g], high_rank[g] - low_rank[g]) + 1 for g in range(len(flags))
    )


def road_gap(stacks: Stacks, size: int, color: Color) -> int:
    """Roughly how many more squares ``color`` needs to complete a road.

    The cheapest path between opposite edges, where squares ``color`` already
    has count 0, empty squares 1, and squares that would have to be taken
    over count more. Returns ``3 * size`` if every path is blocked.
    """
    cost: list[int | None] = []
    for stack in stacks:
        if not stack:
            cost.append(_EMPTY)
            continue
        top = stack[-1]
        if top.color is color:
            cost.append(_OWN_WALL if top.type is PieceType.WALL else _OWN)
        elif top.type is PieceType.CAP:
            cost.append(None)
        else:
            cost.append(_THEIR_WALL if top.type is PieceType.WALL else _THEIRS)
    edges = edges_of(size)
    near = neighbors_of(size)
    best = 3 * size
    for start_edge, end_edge in ((LEFT, RIGHT), (BOTTOM, TOP)):
        distance: dict[int, int] = {}
        queue: list[tuple[int, int]] = []
        for cell, flags in enumerate(edges):
            step = cost[cell]
            if flags & start_edge and step is not None:
                distance[cell] = step
                queue.append((step, cell))
        heapq.heapify(queue)
        while queue:
            spent, cell = heapq.heappop(queue)
            if spent > distance[cell] or spent >= best:
                continue
            if edges[cell] & end_edge:
                best = spent
                break
            for other in near[cell]:
                step = cost[other]
                if step is None:
                    continue
                total = spent + step
                if total < distance.get(other, total + 1):
                    distance[other] = total
                    heapq.heappush(queue, (total, other))
    return best


def features(board: Board, color: Color, stacks: Stacks | None = None) -> Features:
    """Measure ``board`` for ``color``. Pass ``stacks`` if you already have them."""
    size = board.size
    if stacks is None:
        stacks = stacks_of(board)
    weights = centrality(size)
    center = 0.0
    control = flats = captives = 0
    for cell, stack in enumerate(stacks):
        if not stack or stack[-1].color is not color:
            continue
        control += 1
        kind = stack[-1].type
        if kind is PieceType.FLAT:
            flats += 1
        if kind is not PieceType.WALL:
            center += weights[cell]
        captives += sum(1 for piece in stack[:-1] if piece.color is color)
    return Features(
        longest_road=longest_road(stacks, size, color),
        road_gap=road_gap(stacks, size, color),
        center=center,
        control=control,
        flats=flats,
        captives=captives,
    )


def endgame(game: Game, stacks: Stacks | None = None) -> float:
    """How close the game is to ending by filling up or running out: 0.0 early to 1.0."""
    size = game.size
    if stacks is None:
        stacks = stacks_of(game.board)
    empty = sum(1 for stack in stacks if not stack)
    filled = 1 - empty / (size * size)
    lowest = min(reserve.total for reserve in game.reserves.values())
    running_out = 1 - lowest / sum(BOARD_CONFIG[size])
    return max(filled, running_out) ** 2


def evaluate(game: Game, color: Color, weights: Weights = DEFAULT_WEIGHTS) -> float:
    """How good the position is for ``color``: positive is good, negative bad.

    Finished games score far beyond anything else: a win is ``+inf``, a loss
    ``-inf``, and a draw ``0``.
    """
    if game.result is not None:
        if game.result.winner is None:
            return 0.0
        return float("inf") if game.result.winner is color else float("-inf")
    stacks = stacks_of(game.board)
    ours = features(game.board, color, stacks)
    theirs = features(game.board, color.opponent, stacks)
    flat_weight = weights.flats + weights.late_flats * endgame(game, stacks)
    return (
        weights.longest_road * (ours.longest_road - theirs.longest_road)
        - weights.road_gap * (ours.road_gap - theirs.road_gap)
        + weights.center * (ours.center - theirs.center)
        + weights.control * (ours.control - theirs.control)
        + flat_weight * (ours.flats - theirs.flats)
        + weights.captives * (ours.captives - theirs.captives)
    )
