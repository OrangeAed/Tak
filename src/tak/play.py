"""Play Tak against the computer, in the terminal or in the window.

From the command line::

    python -m tak.play                                   # Greedy; you play White on 5x5
    python -m tak.play --bot strategist --color random   # the strongest, either color
    python -m tak.play --bot random --color black --size 4 --gui

Or from Python::

    from tak.play import play_against_bot

    play_against_bot(bot="strategist", color="random", size=6)
    play_against_bot(bot="greedy", color="black", gui=True)

The four computer players, from easiest to hardest, are ``random``,
``greedy``, ``strategist`` and ``learner``; see :mod:`tak.ai`.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from tak.ai import AGENTS, make_agent
from tak.engine.board import BOARD_CONFIG
from tak.engine.game import Game
from tak.engine.pieces import Color
from tak.engine.rules import GameResult
from tak.saves import add_save_arguments, default_saves_dir, saves_dir_from

DEFAULT_BOT = "greedy"
DEFAULT_COLOR = "white"
DEFAULT_SIZE = 5

#: What ``color`` may be: a side, or ``random`` to toss a coin for it.
COLOR_CHOICES = ("white", "black", "random")


def choose_color(choice: str, rng: random.Random | None = None) -> Color:
    """The color the person plays: ``white``, ``black``, or ``random`` for either."""
    choice = choice.lower()
    if choice == "random":
        return (rng or random.Random()).choice(list(Color))
    try:
        return Color(choice)
    except ValueError:
        raise ValueError(
            f"unknown color {choice!r}; choose one of {', '.join(COLOR_CHOICES)}"
        ) from None


def play_against_bot(
    bot: str = DEFAULT_BOT,
    color: str = DEFAULT_COLOR,
    size: int = DEFAULT_SIZE,
    *,
    gui: bool = False,
    seed: int | None = None,
    save_dir: Path | None = None,
    save: bool = True,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> GameResult | None:
    """Play one game against a computer player.

    - ``bot``: which computer player to face: ``random``, ``greedy``, ``strategist``
      or ``learner``.
    - ``color``: the color *you* play: ``white`` (moves first), ``black``, or
      ``random``. The computer plays the other.
    - ``size``: the board size, 3 to 8.
    - ``gui``: play in the 3D window (needs pygame) instead of the terminal.
    - ``seed``: fixes the computer's choices for good (and a random color). In
      any one game the computer always plays the same move in the same
      position; without a seed, a new game may play differently, while with
      the same seed every game does the same.
    - ``save_dir`` / ``save``: where the finished game is saved (default: the
      usual saves folder), or ``save=False`` to not save it.
    - ``read`` / ``write``: how the terminal game gets input and shows output.

    Returns the result in the terminal, or ``None`` if the game was abandoned.
    The window returns ``None`` when it is closed. Raises :class:`ValueError`
    for a bad ``bot``, ``color`` or ``size``.
    """
    if size not in BOARD_CONFIG:
        raise ValueError(f"unsupported board size {size}; choose one of 3 to 8")
    rng = random.Random(seed)
    human = choose_color(color, rng)
    agent = make_agent(bot, seed=rng.randrange(2**32) if seed is not None else None)
    players = {human.opponent: agent}
    folder = (save_dir or default_saves_dir()) if save else None
    picked = " (picked at random)" if color.lower() == "random" else ""
    intro = f"You play {human}{picked} against {agent.name}."

    if gui:
        from tak.gui.app import run as run_window
        from tak.gui.session import INFO, GameSession

        run_window(GameSession(Game(size), folder, players), message=(INFO, intro))
        return None

    from tak.cli.app import run as run_terminal

    write(intro)
    return run_terminal(Game(size), read, write, save_dir=folder, players=players)


def main(
    argv: Sequence[str] | None = None,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    bots = "\n".join(f"  {key:<11} {kind.description}" for key, kind in AGENTS.items())
    parser = argparse.ArgumentParser(
        prog="python -m tak.play",
        description="Play Tak against the computer, in the terminal or in a window.",
        epilog=f"Computer players, easiest first:\n{bots}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--bot",
        choices=list(AGENTS),
        default=DEFAULT_BOT,
        help=f"which computer player to face (default: {DEFAULT_BOT})",
    )
    parser.add_argument(
        "-c",
        "--color",
        type=str.lower,
        choices=COLOR_CHOICES,
        default=DEFAULT_COLOR,
        help=f"the color you play; White moves first (default: {DEFAULT_COLOR})",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        choices=sorted(BOARD_CONFIG),
        default=DEFAULT_SIZE,
        help=f"board size (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="play in the 3D window instead of the terminal (needs pygame)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="fix the computer's choices (and a random color) across games; within one game "
        "it always plays the same move in the same position anyway",
    )
    add_save_arguments(parser)
    args = parser.parse_args(argv)

    if args.gui:
        try:
            import pygame  # noqa: F401
        except ImportError:
            print("The GUI needs pygame. Install it with: pip install -e '.[gui]'", file=sys.stderr)
            return 1

    save_dir = saves_dir_from(args)
    try:
        play_against_bot(
            args.bot,
            args.color,
            args.size,
            gui=args.gui,
            seed=args.seed,
            save_dir=save_dir,
            save=save_dir is not None,
            read=read,
            write=write,
        )
    except KeyboardInterrupt:
        write("")
        write("Game abandoned.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
