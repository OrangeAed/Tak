"""Train the learner by self-play, from scratch, and save its weights.

    python scripts/train_learner.py                    # the weights that come with the game
    python scripts/train_learner.py --games 500        # a quicker, weaker learner
    python scripts/train_learner.py --check 20         # then play 20 games against each bot

Training starts from weights that know nothing and plays the learner against
itself, several games at once (one per CPU core). It never plays the other
computer players; ``--check`` only measures the result against them
afterwards. By default the weights are written to the file the learner reads
(``src/tak/ai/learner_weights.json``), so ``python -m tak.play --bot learner``
plays with them straight away; ``--output`` saves them somewhere else.

Training stops after ``--minutes`` (20 by default) even if it hasn't played
every game, so it never runs longer than that; the weights learned so far are
saved.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tak.ai import LearnerAgent, make_agent
from tak.ai.learning import FEATURE_NAMES, WEIGHTS_PATH, Weights, load_weights, save_weights
from tak.ai.training import Progress, Settings, train
from tak.engine import BOARD_CONFIG, Color, Game

DEFAULTS = Settings()


def show(progress: Progress) -> None:
    print(
        f"{progress.games:>6} games  {progress.seconds:>5.0f}s  "
        f"error {progress.error:.3f}  "
        f"last batch: White {progress.white_wins}, Black {progress.black_wins}, "
        f"draws {progress.draws}, {progress.average_moves:.0f} moves a game",
        flush=True,
    )


def check_game(path: Path, opponent: str, number: int, size: int) -> str:
    """One game between the learner (weights at ``path``) and ``opponent``."""
    learner = LearnerAgent(seed=number, weights=load_weights(path))
    other = make_agent(opponent, seed=10_000 + number)
    learner_color = Color.WHITE if number % 2 == 0 else Color.BLACK
    game = Game(size)
    while not game.is_over and game.ply < 400:
        agent = learner if game.to_move is learner_color else other
        game.play(agent.choose_move(game))
    if game.result is None or game.result.winner is None:
        return "draw"
    return "win" if game.result.winner is learner_color else "loss"


def check(path: Path, games: int, sizes: Sequence[int], workers: int | None) -> None:
    """Play ``games`` games against each of the other computer players, swapping colors."""
    print()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for size in sizes:
            for opponent in ("random", "greedy", "strategist"):
                jobs = [pool.submit(check_game, path, opponent, n, size) for n in range(games)]
                tally = Counter(job.result() for job in jobs)
                print(
                    f"learner vs {opponent} ({size}x{size}, {games} games): "
                    f"{tally['win']} wins, {tally['loss']} losses, {tally['draw']} draws",
                    flush=True,
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-g",
        "--games",
        type=int,
        default=DEFAULTS.games,
        help="self-play games (default %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--sizes",
        type=int,
        nargs="+",
        choices=sorted(BOARD_CONFIG),
        default=list(DEFAULTS.sizes),
        help="board sizes to train on, taking turns (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed, help="makes training repeatable")
    parser.add_argument("--workers", type=int, help="games played at once (default: CPU count)")
    parser.add_argument(
        "--minutes", type=float, default=20, help="stop after this long (default 20)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=WEIGHTS_PATH, help="where to save the weights"
    )
    parser.add_argument(
        "--check", type=int, default=0, metavar="N", help="then play N games against each bot"
    )
    args = parser.parse_args(argv)

    settings = Settings(games=args.games, sizes=tuple(args.sizes), seed=args.seed)
    boards = ", ".join(f"{size}x{size}" for size in settings.sizes)
    print(f"Training on {boards} for up to {args.games} games of self-play.")
    start = time.perf_counter()
    weights: Weights = train(
        settings, workers=args.workers, report=show, time_limit=args.minutes * 60
    )
    minutes = (time.perf_counter() - start) / 60
    save_weights(weights, args.output)
    print(f"\nTrained on {weights.games} games in {minutes:.1f} minutes; saved to {args.output}")
    print("What it learned (a positive weight means more of it is good for the player):")
    for name, weight in zip(FEATURE_NAMES, weights.weights, strict=True):
        print(f"  {name:<14} {weight:+.3f}")
    if args.check:
        check(args.output, args.check, settings.sizes, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
