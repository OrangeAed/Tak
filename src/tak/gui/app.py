"""The graphical game window: a pygame render of the board, reserves, and labels.

Run it with ``tak-gui`` or ``python -m tak.gui``, with ``--size`` choosing the
board (3 to 8). The window shows the board with ranks and files labelled, every
stack on it, and both players' remaining pieces. Moves are typed in PTN, just
like the terminal interface, into the input line at the bottom of the panel.

All rules come from the engine; this module only draws and collects input.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tak.engine.board import BOARD_CONFIG
from tak.engine.game import Game
from tak.engine.notation import NotationError, format_move, parse_move
from tak.engine.pieces import Color
from tak.engine.rules import IllegalMoveError

DEFAULT_SIZE = 5

#: Message kinds, used to pick a color when drawing.
INFO, GOOD, BAD = "info", "good", "bad"


def threat_call(game: Game, mover: Color) -> str | None:
    """What to announce after ``mover``'s move: "Tinue!", "Tak!", or nothing."""
    if not game.threatens_win(mover):
        return None
    return "Tinue!" if game.is_tinue(mover) else "Tak!"


def submit_move(game: Game, text: str) -> tuple[str, str] | None:
    """Try to play ``text`` as a PTN move; return a ``(kind, message)`` to show.

    Returns ``None`` for quiet success, a ``(GOOD, ...)`` announcement for a
    Tak or Tinue call, and a ``(BAD, ...)`` explanation when the text is not
    valid notation or not a legal move.
    """
    mover = game.to_move
    try:
        game.play(parse_move(text))
    except NotationError as exc:
        return BAD, str(exc)
    except IllegalMoveError as exc:
        return BAD, f"Illegal move: {exc}."
    call = threat_call(game, mover)
    return (GOOD, call) if call else None


def run(game: Game) -> None:
    """Open the window and play ``game`` until it ends or the window is closed."""
    import pygame

    from tak.gui import render

    pygame.init()
    layout = render.layout_for(game.size)
    screen = pygame.display.set_mode(layout.window_size)
    pygame.display.set_caption(f"Tak {game.size}\u00d7{game.size}")
    fonts = render.Fonts.default()
    clock = pygame.time.Clock()

    colors = {INFO: render.TEXT, GOOD: render.ACCENT, BAD: render.ERROR}
    typed = ""
    message: tuple[str, str] | None = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                ctrl = event.mod & pygame.KMOD_CTRL
                if ctrl and event.key == pygame.K_q:
                    running = False
                elif ctrl and event.key == pygame.K_z:
                    if game.history:
                        message = INFO, f"Took back {format_move(game.undo())}."
                    else:
                        message = BAD, "There is nothing to undo."
                elif event.key == pygame.K_ESCAPE:
                    typed = ""
                elif event.key == pygame.K_BACKSPACE:
                    typed = typed[:-1]
                elif event.key == pygame.K_RETURN:
                    if typed.strip():
                        message = submit_move(game, typed.strip())
                    typed = ""
                elif event.unicode and event.unicode.isprintable() and not ctrl:
                    typed += event.unicode

        render.draw_game(
            screen,
            game,
            layout,
            fonts,
            message=message[1] if message else None,
            message_color=colors[message[0]] if message else render.TEXT,
            input_text=None if game.is_over else typed,
        )
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tak-gui",
        description="Play Tak in a window, entering PTN moves for both sides.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        choices=sorted(BOARD_CONFIG),
        default=DEFAULT_SIZE,
        help=f"board size (default: {DEFAULT_SIZE})",
    )
    args = parser.parse_args(argv)
    try:
        import pygame  # noqa: F401
    except ImportError:
        print("The GUI needs pygame. Install it with: pip install -e '.[gui]'")
        return 1
    run(Game(args.size))
    return 0
