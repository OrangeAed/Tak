"""The terminal game loop: type PTN moves for both players."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path

from tak.cli.render import LEGEND, describe_result, render_board, render_status
from tak.engine.board import BOARD_CONFIG
from tak.engine.game import Game
from tak.engine.notation import NotationError, format_move, game_to_ptn, parse_move
from tak.engine.pieces import Color
from tak.engine.rules import GameResult, IllegalMoveError
from tak.saves import add_save_arguments, save_finished_game, saves_dir_from

DEFAULT_SIZE = 5

HELP = """\
Enter moves in PTN, the standard Tak notation:
  a1        place a flat stone on a1
  Sb2       place a standing stone (wall) on b2
  Cc3       place a capstone on c3
  c3>       move the top stone of c3 one square right
  3c3>12    pick up 3 stones from c3, move right, drop 1 then 2
Directions are < (left), > (right), + (up), - (down), as seen with a1 at the bottom left.
On each player's first turn you place a flat stone of the OPPONENT's color.

Commands: help, undo, moves (list legal moves), ptn (show the game so far), quit"""


def _prompt(game: Game) -> str:
    mover = game.to_move
    if game.in_opening:
        return f"{mover}, place a {mover.opponent.value} flat stone for your opponent (e.g. a1) > "
    return f"{mover} > "


def threat_call(game: Game, mover: Color) -> str | None:
    """What to announce after ``mover``'s move: "Tinue!", "Tak!", or nothing.

    "Tak!" means the move threatens to win next turn; "Tinue!" means the
    opponent has no reply that stops it.
    """
    if not game.threatens_win(mover):
        return None
    return "Tinue!" if game.is_tinue(mover) else "Tak!"


def run(
    game: Game,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    save_dir: Path | None = None,
) -> GameResult | None:
    """Play ``game`` until someone wins. Returns the result, or ``None`` if play was abandoned.

    If ``save_dir`` is given, a finished game is saved there as a new PTN file.
    Abandoned games are not saved.
    """
    write(f"Tak on a {game.size}x{game.size} board. Type 'help' for how to enter moves.")
    write(LEGEND)

    # "Tak!" or "Tinue!" if the move just played threatens to win next turn. It
    # is announced once, in the status shown straight after that move.
    call: str | None = None
    while game.result is None:
        write("")
        write(render_board(game.board))
        write(render_status(game, call))
        call = None
        try:
            line = read(_prompt(game)).strip()
        except EOFError:
            write("")
            write("Input ended before the game finished.")
            return None
        if not line:
            continue

        command = line.lower()
        if command in {"quit", "exit", "q"}:
            write("Game abandoned.")
            return None
        if command in {"help", "h", "?"}:
            write(HELP)
        elif command in {"undo", "u"}:
            if game.history:
                write(f"Took back {format_move(game.undo())}.")
            else:
                write("There is nothing to undo.")
        elif command == "ptn":
            write(game_to_ptn(game).rstrip())
        elif command == "moves":
            moves = " ".join(format_move(move) for move in game.legal_moves())
            write(textwrap.fill(moves, width=78))
        else:
            mover = game.to_move
            try:
                game.play(parse_move(line))
            except NotationError as exc:
                write(str(exc))
            except IllegalMoveError as exc:
                write(f"Illegal move: {exc}.")
            else:
                call = threat_call(game, mover)

    write("")
    write(render_board(game.board))
    write("")
    write(f"Game over, {describe_result(game, game.result)}")
    write(f"Result: {game.result.ptn}")
    if save_dir is not None:
        write(save_finished_game(game, save_dir)[1])
    return game.result


def main(
    argv: Sequence[str] | None = None,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tak.cli",
        description="Play Tak in the terminal, entering moves for both sides.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        choices=sorted(BOARD_CONFIG),
        default=DEFAULT_SIZE,
        help=f"board size (default: {DEFAULT_SIZE})",
    )
    add_save_arguments(parser)
    args = parser.parse_args(argv)
    try:
        run(Game(args.size), read, write, save_dir=saves_dir_from(args))
    except KeyboardInterrupt:
        write("")
        write("Game abandoned.")
        return 130
    return 0
