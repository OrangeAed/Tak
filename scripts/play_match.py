"""Pit two computer players against each other and report the score.

    python scripts/play_match.py strategist greedy
    python scripts/play_match.py greedy random --games 50 --size 6 --seed 1

The players swap colors every game, so neither gets White's first move in
every game. Games that run past ``--max-moves`` are stopped and counted as
unfinished.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence

from tak.ai import AGENTS, make_agent
from tak.engine import BOARD_CONFIG, Color, Game


def play_game(
    white: str, black: str, size: int, seed: int | None, max_moves: int
) -> tuple[Game, dict[Color, list[float]]]:
    """One game; returns it and how long each side took per move, in seconds."""
    agents = {
        Color.WHITE: make_agent(white, None if seed is None else seed),
        Color.BLACK: make_agent(black, None if seed is None else seed + 1),
    }
    times: dict[Color, list[float]] = {color: [] for color in Color}
    game = Game(size)
    while not game.is_over and game.ply < max_moves:
        mover = game.to_move
        start = time.perf_counter()
        move = agents[mover].choose_move(game)
        times[mover].append(time.perf_counter() - start)
        game.play(move)
    return game, times


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("first", choices=list(AGENTS), help="one computer player")
    parser.add_argument("second", choices=list(AGENTS), help="the other")
    parser.add_argument("-g", "--games", type=int, default=20, help="games to play (default 20)")
    parser.add_argument("-s", "--size", type=int, choices=sorted(BOARD_CONFIG), default=5)
    parser.add_argument("--seed", type=int, help="make the games repeatable")
    parser.add_argument("--max-moves", type=int, default=400, help="stop a game after this many")
    args = parser.parse_args(argv)

    tally: Counter[str] = Counter()
    thinking: dict[str, list[float]] = {"first": [], "second": []}
    for number in range(args.games):
        first_is_white = number % 2 == 0
        white, black = (args.first, args.second) if first_is_white else (args.second, args.first)
        seed = None if args.seed is None else args.seed + 2 * number
        game, times = play_game(white, black, args.size, seed, args.max_moves)
        first_color = Color.WHITE if first_is_white else Color.BLACK
        thinking["first"] += times[first_color]
        thinking["second"] += times[first_color.opponent]
        result = game.result
        if result is None:
            outcome = "unfinished"
        elif result.winner is None:
            outcome = "draw"
        else:
            side = "first" if result.winner is first_color else "second"
            outcome = f"{side} by {result.reason.value}"
        tally[outcome] += 1
        print(
            f"game {number + 1:>3}: {white} (White) vs {black} (Black): "
            f"{result.ptn if result else 'unfinished'} after {game.ply} moves",
            flush=True,
        )

    first_wins = tally["first by road"] + tally["first by flat"]
    second_wins = tally["second by road"] + tally["second by flat"]
    print()
    print(f"{args.size}x{args.size}, {args.games} games")
    print(f"  {args.first}: {first_wins} wins ({tally['first by road']} by road)")
    print(f"  {args.second}: {second_wins} wins ({tally['second by road']} by road)")
    print(f"  draws: {tally['draw']}, unfinished: {tally['unfinished']}")
    for key, label in (("first", args.first), ("second", args.second)):
        spent = thinking[key]
        if spent:
            average, worst = 1000 * sum(spent) / len(spent), 1000 * max(spent)
            print(f"  {label} ({key}) per move: {average:.0f} ms average, {worst:.0f} ms max")
    return 0


if __name__ == "__main__":
    sys.exit(main())
