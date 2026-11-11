"""The graphical game window: a 3D pygame render of the board, reserves, and labels.

Run it with ``tak-gui`` or ``python -m tak.gui``, with ``--size`` choosing the
board (3 to 8). The window shows the board in 3D with ranks and files labelled,
every stack on it built up piece by piece, and both players' remaining pieces.
Drag with the mouse or hold the arrow keys to turn and tilt the board, scroll
to zoom, Tab to look from the other side, and Home to reset the view. Moves are
typed in PTN, just like the terminal interface, into the input line in the panel.

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


#: How fast the board turns: radians per pixel dragged, and per second a key is held.
DRAG_SPEED = 0.008
KEY_SPEED = 1.6


def run(game: Game) -> None:
    """Open the window and play ``game`` until it ends or the window is closed."""
    import pygame

    from tak.gui import render
    from tak.gui.scene import Camera

    pygame.init()
    layout = render.layout_for(game.size)
    screen = pygame.display.set_mode(layout.window_size)
    pygame.display.set_caption(f"Tak {game.size}\u00d7{game.size}")
    fonts = render.Fonts.default()
    clock = pygame.time.Clock()

    colors = {INFO: render.TEXT, GOOD: render.ACCENT, BAD: render.ERROR}
    typed = ""
    message: tuple[str, str] | None = None
    camera = Camera()
    dragging = False
    dirty = True  # a 3D frame is costly, so only redraw when something changed

    running = True
    while running:
        seconds = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = layout.viewport.collidepoint(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                camera = camera.orbit(-dx * DRAG_SPEED, dy * DRAG_SPEED)
            elif event.type == pygame.MOUSEWHEEL:
                camera = camera.zoomed(1.1**event.y)
            elif event.type == pygame.KEYDOWN:
                ctrl = event.mod & pygame.KMOD_CTRL
                if ctrl and event.key == pygame.K_q:
                    running = False
                elif ctrl and event.key == pygame.K_z:
                    if game.history:
                        message = INFO, f"Took back {format_move(game.undo())}."
                    else:
                        message = BAD, "There is nothing to undo."
                elif event.key == pygame.K_TAB:
                    camera = camera.flipped()
                elif event.key == pygame.K_HOME:
                    camera = Camera()
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
            else:
                continue  # nothing that changes the picture
            dirty = True

        # Held arrow keys turn (left/right) and tilt (up/down) the board smoothly.
        held = pygame.key.get_pressed()
        turn = held[pygame.K_LEFT] - held[pygame.K_RIGHT]
        tilt = held[pygame.K_UP] - held[pygame.K_DOWN]
        if turn or tilt:
            camera = camera.orbit(turn * KEY_SPEED * seconds, tilt * KEY_SPEED * seconds)
            dirty = True

        if dirty:
            render.draw_game(
                screen,
                game,
                layout,
                fonts,
                camera=camera,
                message=message[1] if message else None,
                message_color=colors[message[0]] if message else render.TEXT,
                input_text=None if game.is_over else typed,
            )
            pygame.display.flip()
            dirty = False

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
