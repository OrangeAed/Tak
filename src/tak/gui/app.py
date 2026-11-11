"""The graphical window: shows a game of Tak drawn with pygame.

For now this only displays the game; moves can't be made in the window yet.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from tak.engine.board import BOARD_CONFIG
from tak.engine.game import Game

DEFAULT_SIZE = 5
WINDOW_SIZE = (1040, 720)
INSTALL_HINT = 'The GUI needs pygame. Install it with:  pip install -e ".[gui]"'


def run(game: Game, window_size: tuple[int, int] = WINDOW_SIZE) -> None:
    """Open a window showing ``game`` until it is closed (or Escape is pressed).

    The window can be resized; the board is redrawn to fit.
    """
    import pygame

    from tak.gui.render import draw_game

    pygame.display.init()  # only what's needed; audio in particular isn't
    pygame.font.init()
    try:
        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption(f"Tak ({game.size}x{game.size})")
        while True:
            draw_game(screen, game)
            pygame.display.flip()
            event = pygame.event.wait()  # nothing changes between events, so just wait
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.get_surface()
    finally:
        pygame.quit()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tak.gui",
        description="Show a Tak board in a window.",
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

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # pygame's import banner
    try:
        import pygame  # noqa: F401
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1

    try:
        run(Game(args.size))
    except KeyboardInterrupt:
        return 130
    return 0
