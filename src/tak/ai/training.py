"""Teaching the learner to play: temporal-difference learning from self-play.

The learner starts knowing nothing but the rules and a few sure-fire tactics
(take a win, block a win, spot Tinue). Its evaluation starts with every
weight at zero, so every quiet move looks as good as any other, and it loses
nearly every game against the greedy player and the strategist. Then it plays
itself, over and over, and after each game nudges its weights so that its
opinion of each position moves toward what it thought of the *next* position,
and, at the end, toward how the game actually ended. This is TD(lambda)
learning, the method behind TD-Gammon.

Over a thousand games or so, the weights settle on what wins: which of road
building, flat count, blocking, center control, and so on matter, and how
much. Nothing here looks at how the other computer players play; the learner
only ever plays itself.

Games are played in batches, several at once on separate processes, all with
the same weights; then the weights are updated from each game of the batch in
order. Each game's randomness comes from the training seed and the game's
number alone, so the same seed and settings give the same weights however
many processes are used.

Run it with ``python scripts/train_learner.py`` (``--help`` for the options).
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

from tak.ai.base import opening_move
from tak.ai.learner_agent import LearnerAgent, find_win
from tak.ai.learning import Weights, value
from tak.engine.game import Game
from tak.engine.pieces import Color


@dataclass(frozen=True)
class Settings:
    """How to train. The defaults finish in well under 20 minutes on a 4-core laptop."""

    #: Self-play games to play in all.
    games: int = 1000
    #: Board sizes to train on, taking turns. The features are scaled by
    #: size, so weights trained on 5x5 play well on the other sizes too.
    sizes: tuple[int, ...] = (5,)
    #: How far each update moves the weights, at the start and at the end
    #: (it shrinks steadily in between).
    learning_rate: float = 0.02
    final_learning_rate: float = 0.002
    #: TD(lambda): how far back along the game each result reaches (0 to 1).
    lam: float = 0.7
    #: Chance of a random (safe) move instead of the best one, to try new things.
    explore: float = 0.05
    #: Each game starts with up to this many random safe moves, so games differ.
    random_opening: int = 6
    #: How many moves to check for letting the opponent make Tinue (see
    #: :attr:`LearnerAgent.tinue_checks`). The check only changes which move
    #: is played, not how positions are judged, and skipping it in training
    #: makes training about twice as fast.
    tinue_checks: int = 0
    #: Games that run longer than this are stopped and count as draws.
    max_moves: int = 200
    #: Games played at once, all with the same weights, before updating.
    batch: int = 16
    #: The weights returned are the average of the weights after each game
    #: from this far through training on (0.5: the second half). Weights from
    #: any one moment wobble from game to game; their average is steadier.
    average_from: float = 0.5
    #: Makes training repeatable.
    seed: int = 0


@dataclass
class Trajectory:
    """What training needs from one self-play game."""

    #: The features of every position reached (after the opening), from the
    #: point of view of the player who had just moved, and whether that was White.
    positions: list[tuple[list[float], bool]]
    #: The winner, or ``None`` for a draw (or a game stopped for running long).
    winner: Color | None
    moves: int


def self_play(weights: Weights, settings: Settings, number: int) -> Trajectory:
    """Play game ``number`` of a training run, the learner against itself."""
    rng = random.Random(f"{settings.seed}|{number}")
    agent = LearnerAgent(seed=0, weights=weights, tinue_checks=settings.tinue_checks)
    game = Game(settings.sizes[number % len(settings.sizes)])
    positions: list[tuple[list[float], bool]] = []
    opening = rng.randint(0, settings.random_opening)
    while not game.is_over and game.ply < settings.max_moves:
        if game.in_opening:
            game.play(opening_move(game, rng))
            continue
        me = game.to_move
        win = find_win(game, me)
        if win is not None:
            game.play(win)
            break
        options = agent.consider(game, rng)
        choice = None
        if game.ply < 2 + opening or rng.random() < settings.explore:
            safe = [option for option in options if option.safe]
            choice = rng.choice(safe) if safe else None
        if choice is None:
            choice = agent.best(game, options)
        game.play(choice.move)
        if not game.is_over and choice.features is not None:
            positions.append((choice.features, me is Color.WHITE))
    winner = game.result.winner if game.result is not None else None
    return Trajectory(positions, winner, game.ply)


def td_update(weights: Weights, game: Trajectory, learning_rate: float, lam: float) -> float:
    """Move ``weights`` toward TD(lambda) targets for one game. Returns the mean squared error.

    Values are worked out with the weights as they were when the game
    started (the "offline" lambda-return), from White's point of view so that
    each position can be compared with the next one, which is the other
    player's.
    """
    if not game.positions:
        return 0.0
    outcome = 0.0 if game.winner is None else 1.0 if game.winner is Color.WHITE else -1.0
    # For White: +1 where White had just moved (the value is White's), else -1.
    signs = [1.0 if white else -1.0 for _, white in game.positions]
    values = [sign * value(weights, x) for (x, _), sign in zip(game.positions, signs, strict=True)]

    # Lambda-returns, from the end back: blend the next position's value with
    # the return that followed it.
    targets = [0.0] * len(values)
    following = outcome
    for t in range(len(values) - 1, -1, -1):
        targets[t] = following
        following = (1 - lam) * values[t] + lam * following

    error_total = 0.0
    for (x, _), sign, estimate, target in zip(game.positions, signs, values, targets, strict=True):
        # The error from the mover's point of view. Using it without tanh's
        # slope (as in logistic regression) keeps confident mistakes learning fast.
        error = sign * (target - estimate)
        error_total += error * error
        step = learning_rate * error
        weights.bias += step * _BIAS_SCALE
        for i, feature in enumerate(x):
            weights.weights[i] += step * feature
    weights.games += 1
    return error_total / len(values)


#: The bias (how good it is, in general, to have just moved) learns more slowly
#: than the weights, since every position has it.
_BIAS_SCALE = 0.1


@dataclass
class Progress:
    """Reported after each batch of games."""

    games: int
    seconds: float
    error: float
    white_wins: int
    black_wins: int
    draws: int
    average_moves: float


@contextmanager
def _pool(workers: int) -> Iterator[Executor | None]:
    if workers <= 1:
        yield None
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield pool


def train(
    settings: Settings = Settings(),
    weights: Weights | None = None,
    workers: int | None = None,
    report: Callable[[Progress], None] | None = None,
    time_limit: float | None = None,
) -> Weights:
    """Train by self-play and return the weights.

    - ``weights``: where to start (default: all zeros, knowing nothing).
    - ``workers``: processes playing games at once (default: one per CPU).
    - ``report``: called after every batch, e.g. to print progress.
    - ``time_limit``: stop after this many seconds (after the current batch),
      even if fewer than ``settings.games`` games have been played.

    The result is the average of the weights over the later part of training
    (see :attr:`Settings.average_from`).
    """
    weights = weights.copy() if weights is not None else Weights.zeros()
    weights.sizes = settings.sizes
    workers = workers or os.cpu_count() or 1
    start = time.perf_counter()
    played = 0
    average = _Average()
    average_from = settings.average_from * settings.games
    with _pool(workers) as pool:
        while played < settings.games:
            numbers = range(played, min(played + settings.batch, settings.games))
            frozen = [weights.copy()] * len(numbers)
            runs = [settings] * len(numbers)
            if pool is None:
                games = list(map(self_play, frozen, runs, numbers))
            else:
                games = list(pool.map(self_play, frozen, runs, numbers))
            errors = []
            for game in games:
                progress = played / max(settings.games - 1, 1)
                rate = settings.learning_rate + progress * (
                    settings.final_learning_rate - settings.learning_rate
                )
                errors.append(td_update(weights, game, rate, settings.lam))
                played += 1
                if played > average_from:
                    average.add(weights)
            if report is not None:
                report(
                    Progress(
                        games=played,
                        seconds=time.perf_counter() - start,
                        error=sum(errors) / len(errors),
                        white_wins=sum(g.winner is Color.WHITE for g in games),
                        black_wins=sum(g.winner is Color.BLACK for g in games),
                        draws=sum(g.winner is None for g in games),
                        average_moves=sum(g.moves for g in games) / len(games),
                    )
                )
            if time_limit is not None and time.perf_counter() - start >= time_limit:
                break
    return average.result(weights)


class _Average:
    """A running average of weights."""

    def __init__(self) -> None:
        self.count = 0
        self.weights: list[float] = []
        self.bias = 0.0

    def add(self, weights: Weights) -> None:
        if not self.count:
            self.weights = [0.0] * len(weights.weights)
        self.count += 1
        self.bias += weights.bias
        for i, weight in enumerate(weights.weights):
            self.weights[i] += weight

    def result(self, latest: Weights) -> Weights:
        """The average, recorded as trained on as many games as ``latest``; or ``latest``."""
        if not self.count:
            return latest
        return Weights(
            [total / self.count for total in self.weights],
            self.bias / self.count,
            latest.games,
            latest.sizes,
        )
