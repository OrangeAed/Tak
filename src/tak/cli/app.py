"""The terminal game loop: type PTN moves for both players."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from tak.ai.assess import add_eval_bar_argument, assess
from tak.ai.base import Agent, player_headers
from tak.cli.render import LEGEND, describe_result, render_board, render_status, with_eval_bar
from tak.engine.board import BOARD_CONFIG
from tak.engine.game import Game
from tak.engine.moves import Move
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

HELP_AGAINST_COMPUTER = """\
The computer moves as soon as it is its turn. Against the computer, undo takes
back your last move and the computer's reply, so it is your turn again."""


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
    players: Mapping[Color, Agent] | None = None,
    eval_bar: bool = False,
) -> GameResult | None:
    """Play ``game`` until someone wins. Returns the result, or ``None`` if play was abandoned.

    ``players`` gives the computer player for either or both sides; moves for
    the other sides are read from the person at the terminal. If ``save_dir``
    is given, a finished game is saved there as a new PTN file. Abandoned
    games are not saved. ``eval_bar`` shows an evaluation bar beside the
    board, with a line saying who is ahead (see :mod:`tak.ai.assess`).
    """
    players = dict(players or {})
    write(f"Tak on a {game.size}x{game.size} board. Type 'help' for how to enter moves.")
    for color, agent in players.items():
        write(f"{color} is played by the computer: {agent.name}, which {agent.description}.")
    write(LEGEND)

    # "Tak!" or "Tinue!" if the move just played threatens to win next turn. It
    # is announced once, in the status shown straight after that move.
    call: str | None = None
    while game.result is None:
        computer = players.get(game.to_move)
        if computer is not None:
            if call:
                write(call)  # about the move just typed; the computer answers straight away
            mover = game.to_move
            opening = game.in_opening
            game.play(computer.choose_move(game))
            played = game.history[-1]
            move_text = format_move(played.move, flattened=played.flattened)
            if opening:
                other = players.get(mover.opponent)
                whom = other.name if other else "you"
                flat = f"a {mover.opponent.value} flat stone"
                write(f"{mover} ({computer.name}) places {flat} for {whom} on {move_text}.")
            else:
                write(f"{mover} ({computer.name}) plays {move_text}.")
            call = threat_call(game, mover)
            continue

        write("")
        if eval_bar:
            assessment = assess(game)
            write(with_eval_bar(render_board(game.board), assessment))
            write(render_status(game, call, assessment.detail))
        else:
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
            if players:
                write(HELP_AGAINST_COMPUTER)
        elif command in {"undo", "u"}:
            taken = _undo(game, players)
            if taken:
                write(f"Took back {' and '.join(format_move(move) for move in taken)}.")
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
    board = render_board(game.board)
    write(with_eval_bar(board, assess(game)) if eval_bar else board)
    write("")
    write(f"Game over, {describe_result(game, game.result)}")
    write(f"Result: {game.result.ptn}")
    verdict = _verdict(game.result, players)
    if verdict:
        write(verdict)
    if save_dir is not None:
        write(save_finished_game(game, save_dir, headers=player_headers(players))[1])
    return game.result


def _mover(ply: int) -> Color:
    """Who made the move numbered ``ply`` (counting from 0)."""
    return Color.WHITE if ply % 2 == 0 else Color.BLACK


def _undo(game: Game, players: Mapping[Color, Agent]) -> list[Move]:
    """Take back the person's last move, and any computer moves made since.

    Returns the moves taken back, earliest first; empty if the person hasn't
    moved yet. Without computer players, this is simply the last move.
    """
    if all(_mover(ply) in players for ply in range(game.ply)):
        return []
    taken: list[Move] = []
    while True:
        by_person = _mover(game.ply - 1) not in players
        taken.append(game.undo())
        if by_person:
            break
    taken.reverse()
    return taken


def _verdict(result: GameResult, players: Mapping[Color, Agent]) -> str | None:
    """The last word when a person played the computer: who won, if anyone."""
    if not players or len(players) == len(Color) or result.winner is None:
        return None
    if result.winner in players:
        return f"{players[result.winner].name} wins."
    return "You win!"


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
    add_eval_bar_argument(parser)
    add_save_arguments(parser)
    args = parser.parse_args(argv)
    try:
        run(Game(args.size), read, write, save_dir=saves_dir_from(args), eval_bar=args.eval_bar)
    except KeyboardInterrupt:
        write("")
        write("Game abandoned.")
        return 130
    return 0
