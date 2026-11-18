"""A position evaluation that is learned, not tuned by hand.

The learner (:class:`tak.ai.learner_agent.LearnerAgent`) judges a move by the
position it leaves, the same way the strategist does. The difference is where
the judgement comes from. The strategist adds up a few measurements with
weights someone picked by hand. The learner measures more things and weighs
them with numbers it worked out itself by playing against itself (see
:mod:`tak.ai.training`).

The value of a position is a number between -1 (lost) and +1 (won) for the
player who just moved::

    value = tanh(bias + sum(weight[i] * feature[i]))

Each feature is a simple count or measurement of the board (listed in
:data:`FEATURE_NAMES`), nearly all of them "ours minus theirs". They are
scaled by the board size so that weights learned on one size make sense on
the others. This is a linear model: small, quick to evaluate, and quick to
train, which matters because the engine is plain Python.

The weights live in a small JSON file, ``learner_weights.json``, next to this
module; :func:`load_weights` reads it and :func:`save_weights` writes it.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tak.ai.evaluation import centrality, endgame, road_gap
from tak.ai.tactics import (
    Stacks,
    edges_of,
    groups,
    neighbors_of,
    road_cells,
    spans,
    stacks_of,
)
from tak.engine.board import BOARD_CONFIG
from tak.engine.game import Game
from tak.engine.pieces import Color, PieceType

#: Where the trained weights are kept.
WEIGHTS_PATH = Path(__file__).with_name("learner_weights.json")

#: What each feature measures, in order. All are from the point of view of the
#: player who just moved ("we"), against the player about to move ("they").
FEATURE_NAMES: tuple[str, ...] = (
    # Flat stones on top: what decides a flat win.
    "flats",
    # The same, but only counting near the end of the game, when it matters most.
    "flats_late",
    # Walls on top: they block roads but don't count as flats.
    "walls",
    # Capstones on the board: they block, flatten walls and can't be covered.
    "caps",
    # Squares controlled, weighted toward the middle of the board.
    "center",
    # Our own stones buried in stacks we control (reserves we can spread out).
    "captives",
    # Their stones buried under our pieces (they can't use them while we hold them).
    "prisoners",
    # How many squares short of a road each player is; theirs minus ours.
    "road_gap",
    # How far across the board the best connected group reaches.
    "longest_road",
    # Road pieces lined up in the same rank or file (lines with more count more).
    "lines",
    # Empty squares where a stone would complete a road right away.
    "road_squares",
    # Pieces left in reserve.
    "reserves",
    # The rest are not ours minus theirs. Whether we threaten to win on our
    # next move (Tak).
    "tak",
    # Whether we are exactly two or three squares short of a road (by
    # "road_gap"). Two short means we could make Tak next move, maybe two at
    # once; these let the evaluation treat being close to a road as more than
    # a straight sum of squares.
    "our_gap_2",
    "our_gap_3",
    # The same for them. They move next, so these matter even more.
    "their_gap_2",
    "their_gap_3",
)

NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class Weights:
    """The learned numbers: one weight per feature, and a bias."""

    weights: list[float]
    bias: float = 0.0
    #: How many self-play games trained these (just for the record).
    games: int = 0
    #: The board sizes they were trained on (just for the record).
    sizes: tuple[int, ...] = ()

    @classmethod
    def zeros(cls) -> Weights:
        """Weights that know nothing: every position looks even."""
        return cls([0.0] * NUM_FEATURES)

    def copy(self) -> Weights:
        return Weights(list(self.weights), self.bias, self.games, self.sizes)

    def to_json(self) -> str:
        data = {
            "features": dict(zip(FEATURE_NAMES, self.weights, strict=True)),
            "bias": self.bias,
            "games": self.games,
            "sizes": list(self.sizes),
        }
        return json.dumps(data, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Weights:
        data = json.loads(text)
        named = data["features"]
        missing = [name for name in FEATURE_NAMES if name not in named]
        if missing:
            raise ValueError(f"weights are missing features: {', '.join(missing)}")
        return cls(
            [float(named[name]) for name in FEATURE_NAMES],
            float(data.get("bias", 0.0)),
            int(data.get("games", 0)),
            tuple(int(size) for size in data.get("sizes", ())),
        )


def load_weights(path: Path | None = None) -> Weights:
    """The weights saved at ``path`` (by default, the ones that come with the game)."""
    return Weights.from_json((path or WEIGHTS_PATH).read_text(encoding="utf-8"))


def save_weights(weights: Weights, path: Path | None = None) -> None:
    (path or WEIGHTS_PATH).write_text(weights.to_json(), encoding="utf-8")


@functools.cache
def default_weights() -> Weights:
    """The weights that come with the game, read once."""
    return load_weights()


def value(weights: Weights, x: Sequence[float]) -> float:
    """How good a position with features ``x`` is for the player who just moved, -1 to +1."""
    total = weights.bias
    for weight, feature in zip(weights.weights, x, strict=True):
        total += weight * feature
    return math.tanh(total)


# --- Features ------------------------------------------------------------------------


@functools.cache
def _lines_of(size: int) -> tuple[tuple[int, ...], ...]:
    """Every rank and file, as square indices."""
    ranks = [tuple(rank * size + file for file in range(size)) for rank in range(size)]
    files = [tuple(rank * size + file for rank in range(size)) for file in range(size)]
    return tuple(ranks + files)


def _road_squares(stacks: Stacks, size: int, cells: list[bool]) -> int:
    """Empty squares where placing a flat would complete a road for the owner of ``cells``."""
    label, flags = groups(cells, size)
    if not flags:
        return 0
    edges = edges_of(size)
    near = neighbors_of(size)
    count = 0
    for cell, stack in enumerate(stacks):
        if stack:
            continue
        touched = edges[cell]
        for other in near[cell]:
            if label[other] >= 0:
                touched |= flags[label[other]]
        if spans(touched):
            count += 1
    return count


def _lines(size: int, cells: list[bool], blockers: list[bool]) -> float:
    """How well road pieces line up: each rank or file scores its count squared.

    A line the opponent has a piece in (``blockers``) scores half.
    """
    total = 0.0
    for line in _lines_of(size):
        count = sum(1 for cell in line if cells[cell])
        if count > 1:
            blocked = any(blockers[cell] for cell in line)
            total += count * count * (0.5 if blocked else 1.0)
    return total


def features(game: Game, me: Color, threat: bool, stacks: Stacks | None = None) -> list[float]:
    """Measure ``game``'s position for ``me``, the player who just moved.

    ``threat`` is whether ``me`` could win with their next move (which the
    caller has usually worked out already). See :data:`FEATURE_NAMES`.
    """
    size = game.size
    if stacks is None:
        stacks = stacks_of(game.board)
    them = me.opponent
    center_weights = centrality(size)

    flats = walls = caps = captives = prisoners = 0
    center = 0.0
    for cell, stack in enumerate(stacks):
        if not stack:
            continue
        top = stack[-1]
        sign = 1 if top.color is me else -1
        kind = top.type
        if kind is PieceType.FLAT:
            flats += sign
        elif kind is PieceType.WALL:
            walls += sign
        else:
            caps += sign
        if kind is not PieceType.WALL:
            center += sign * center_weights[cell]
        if len(stack) > 1:
            owner = top.color
            own = sum(1 for piece in stack[:-1] if piece.color is owner)
            captives += sign * own
            prisoners += sign * (len(stack) - 1 - own)

    mine = road_cells(stacks, me)
    theirs = road_cells(stacks, them)
    span_mine, span_theirs = _span(mine, size), _span(theirs, size)
    our_gap, their_gap = road_gap(stacks, size, me), road_gap(stacks, size, them)
    stones, capstones = BOARD_CONFIG[size]
    reserves = game.reserve(me).total - game.reserve(them).total
    late = endgame(game, stacks)

    return [
        flats / size,
        late * flats / size,
        walls / size,
        float(caps),
        center / size,
        captives / size,
        prisoners / size,
        (their_gap - our_gap) / size,
        (span_mine - span_theirs) / size,
        (_lines(size, mine, theirs) - _lines(size, theirs, mine)) / (size * size),
        float(
            min(_road_squares(stacks, size, mine), 2) - min(_road_squares(stacks, size, theirs), 2)
        ),
        reserves / (stones + capstones) * 4,
        1.0 if threat else 0.0,
        float(our_gap == 2),
        float(our_gap == 3),
        float(their_gap == 2),
        float(their_gap == 3),
    ]


def _span(cells: list[bool], size: int) -> int:
    """How many squares along a rank or file the best connected group of ``cells`` covers."""
    label, flags = groups(cells, size)
    if not flags:
        return 0
    low_file = [size] * len(flags)
    high_file = [-1] * len(flags)
    low_rank = [size] * len(flags)
    high_rank = [-1] * len(flags)
    for cell, group in enumerate(label):
        if group < 0:
            continue
        file, rank = cell % size, cell // size
        low_file[group] = min(low_file[group], file)
        high_file[group] = max(high_file[group], file)
        low_rank[group] = min(low_rank[group], rank)
        high_rank[group] = max(high_rank[group], rank)
    return max(
        max(high_file[g] - low_file[g], high_rank[g] - low_rank[g]) + 1 for g in range(len(flags))
    )
