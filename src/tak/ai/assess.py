"""Who is ahead, and by how much, as the learner sees it: the evaluation bar.

The learner's learned evaluation (:mod:`tak.ai.learning`) scores a position
from -1 (lost) to +1 (won) for the player who has just moved. Turned round to
White's point of view, that is what the evaluation bar shows, with two
corrections the evaluation can't see by itself, taken from the same sure-fire
tactics the learner plays by:

- If the player to move can win on the spot, they are winning.
- If the player who just moved has Tinue (a threat that can't be stopped),
  they are winning.

A finished game shows its result. This is the same number the learner gives
in its explanations ("How it rates its position now"), so the bar and the
"Why this move?" card agree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from tak.ai.learner_agent import find_win
from tak.ai.learning import Weights, default_weights, features, value
from tak.ai.tactics import is_tinue
from tak.engine.game import Game
from tak.engine.pieces import Color

#: Beyond this (either way) a side counts as clearly ahead, and beyond
#: :data:`FAR` well ahead; within :data:`NEAR` it's about even.
NEAR = 0.2
FAR = 0.6


@dataclass(frozen=True)
class Assessment:
    """How a position looks, from White's point of view."""

    #: -1 (Black is winning) to +1 (White is winning); 0 is even.
    white: float
    #: Short, for the end of the bar: the score of the side ahead (``0.25``),
    #: ``Win`` for a win that can't be stopped, or the result once the game is over.
    label: str
    #: A sentence, e.g. ``White is slightly ahead (+0.25).``
    detail: str
    #: Whether the game is decided: over, or won whatever the other side plays.
    decided: bool = False

    @property
    def leader(self) -> Color | None:
        """The side ahead, or ``None`` if it's dead even."""
        if self.white > 0:
            return Color.WHITE
        if self.white < 0:
            return Color.BLACK
        return None


def assess(game: Game, weights: Weights | None = None) -> Assessment:
    """How ``game``'s position looks to the learner (with ``weights``, by default its own)."""
    weights = weights if weights is not None else default_weights()
    result = game.result
    if result is not None:
        if result.winner is None:
            return Assessment(0.0, "Draw", f"The game is drawn ({result.ptn}).", True)
        return Assessment(
            _toward(result.winner, 1.0),
            result.ptn,
            f"{result.winner} has won ({result.ptn}).",
            True,
        )

    mover = game.to_move
    just_moved = mover.opponent
    threat = None
    if not game.in_opening:  # nobody can win in the first two moves
        if find_win(game, mover) is not None:
            return Assessment(
                _toward(mover, 1.0), "Win", f"{mover} can win with their next move.", True
            )
        threat = find_win(game, just_moved)
        if threat is not None and is_tinue(game, just_moved, hints=[threat]):
            return Assessment(
                _toward(just_moved, 1.0),
                "Win",
                f"{just_moved} has Tinue: they win whatever {mover} plays.",
                True,
            )
    rating = value(weights, features(game, just_moved, threat=threat is not None))
    white = _toward(just_moved, rating)
    return Assessment(white, f"{abs(white):.2f}", f"{_words(white)} ({white:+.2f}).")


def _toward(color: Color, amount: float) -> float:
    """``amount`` in ``color``'s favor, from White's point of view."""
    return amount if color is Color.WHITE else -amount


def _words(white: float) -> str:
    if abs(white) <= NEAR:
        return "It's about even"
    leader = Color.WHITE if white > 0 else Color.BLACK
    how = "well ahead" if abs(white) > FAR else "ahead"
    return f"{leader} is {how}"


def add_eval_bar_argument(parser: argparse.ArgumentParser) -> None:
    """The ``--eval-bar`` option, shared by the terminal, the window, and ``tak.play``."""
    parser.add_argument(
        "--eval-bar",
        action="store_true",
        help="show an evaluation bar on the left: who is ahead, and by how much, "
        "as the learner sees it",
    )
