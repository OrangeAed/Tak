"""The graphical game window: play Tak, or step through a saved game, in 3D.

Run it with ``python -m tak.gui``: ``--size`` picks the board for a new game
(3 to 8), or ``--load FILE`` opens a saved game to step through, and play on
from any point if you like.

- Click an empty square to place a piece, or your stack to lift it and then
  the squares to drop it on (see :mod:`tak.gui.clicks`). Moves can also be
  typed in PTN, just like the terminal interface.
- Drag with the mouse or hold the arrow keys to turn and tilt the board,
  scroll to zoom, Tab to look from the other side, Home to reset the view.
- Step through the game with the panel's buttons, Ctrl+Left/Right (or
  Ctrl+Z/Y), and Ctrl+Home/End.
- The window can be resized.

All rules come from the engine; this module only draws and collects input.
The event handling lives in :class:`Controller`, which takes one pygame event
at a time, so it can be tested without running a window.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tak.engine.board import BOARD_CONFIG, Square
from tak.engine.game import Game
from tak.engine.notation import format_move
from tak.engine.pieces import PieceType
from tak.gui.clicks import MoveBuilder, touched_squares
from tak.gui.session import BAD, GOOD, INFO, Feedback, GameSession
from tak.saves import add_save_arguments, describe_path, saves_dir_from

DEFAULT_SIZE = 5

#: How fast the board turns: radians per pixel dragged, and per second a key is held.
DRAG_SPEED = 0.008
KEY_SPEED = 1.6
#: A press that moves further than this many pixels is a drag, not a click.
DRAG_THRESHOLD = 5
#: Key repeat: delay before repeating, and time between repeats, in milliseconds.
KEY_REPEAT_DELAY = 400
KEY_REPEAT_INTERVAL = 35


class Controller:
    """Everything the window does in response to input, one event at a time."""

    def __init__(
        self,
        session: GameSession,
        window_size: tuple[int, int] | None = None,
        message: Feedback | None = None,
    ) -> None:
        import pygame

        from tak.gui import render
        from tak.gui.scene import Camera

        self.pg = pygame
        self.render = render
        self.session = session
        self.builder = MoveBuilder()
        self.layout = render.layout_for(session.game.size, window_size or render.DEFAULT_WINDOW)
        self.fonts = render.Fonts.default()
        self.camera = Camera()
        self.default_camera = Camera()

        self.typed = ""
        self.message: Feedback | None = message
        self.place = PieceType.FLAT  # what a plain click on an empty square places
        self.frame: Any = None  # the last frame drawn (a render.Frame), for picking
        self.hover: Square | None = None
        self.held_keys: set[int] = set()  # keys down now, to tell a repeat from a press
        self.press: tuple[int, tuple[int, int]] | None = None  # mouse button and where
        self.dragging = False
        self.running = True
        self.dirty = True  # a 3D frame is costly, so only redraw when something changed

    @property
    def game(self) -> Game:
        return self.session.game

    # --- Events -----------------------------------------------------------------

    def handle(self, event: Any) -> None:
        """React to one pygame event."""
        pg = self.pg
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.MOUSEBUTTONDOWN and event.button in (1, 3):
            self.press = (event.button, event.pos)
            self.dragging = False
        elif event.type == pg.MOUSEMOTION:
            self._mouse_moved(event)
            return  # _mouse_moved decides whether to redraw
        elif event.type == pg.MOUSEBUTTONUP and self.press and event.button == self.press[0]:
            if not self.dragging:
                if event.button == 1:
                    self.click(event.pos, pg.key.get_mods())
                else:
                    self.cancel()
            self.press = None
            self.dragging = False
        elif event.type == pg.MOUSEWHEEL:
            self.camera = self.camera.zoomed(1.1**event.y)
        elif event.type == pg.KEYDOWN:
            repeat = event.key in self.held_keys
            self.held_keys.add(event.key)
            self.key(event, repeat)
        elif event.type == pg.KEYUP:
            self.held_keys.discard(event.key)
            return
        elif event.type in (pg.WINDOWFOCUSLOST, pg.WINDOWLEAVE):
            self.held_keys.clear()  # we won't see the key-ups
            self.press = None
            self.hover = None
        else:
            return
        self.dirty = True

    def _mouse_moved(self, event: Any) -> None:
        if self.press is not None:
            start = self.press[1]
            far = math.dist(event.pos, start) > DRAG_THRESHOLD
            if not self.dragging and far and self.layout.viewport.collidepoint(start):
                self.dragging = True
            if self.dragging:
                dx, dy = event.rel
                self.camera = self.camera.orbit(-dx * DRAG_SPEED, dy * DRAG_SPEED)
                self.dirty = True
        hover = self.frame.square_at(event.pos) if self.frame and not self.dragging else None
        if hover != self.hover:
            self.hover = hover
            self.dirty = True

    def key(self, event: Any, repeat: bool = False) -> None:
        """A key press. ``repeat`` is True for repeats from holding the key down."""
        pg = self.pg
        ctrl = event.mod & pg.KMOD_CTRL
        key = event.key
        if ctrl and key == pg.K_q:
            if not repeat:
                self.running = False
        elif ctrl and key in (pg.K_z, pg.K_LEFT):
            self.step(self.session.back)
        elif ctrl and key in (pg.K_y, pg.K_RIGHT):
            self.step(self.session.forward)
        elif ctrl and key == pg.K_HOME:
            self.step(self.session.to_start)
        elif ctrl and key == pg.K_END:
            self.step(self.session.to_end)
        elif key == pg.K_TAB:
            if not repeat:  # holding Tab shouldn't spin the board
                self.camera = self.camera.flipped()
        elif key == pg.K_HOME:
            self.camera = self.default_camera
        elif key == pg.K_ESCAPE:
            self.cancel()
        elif key == pg.K_BACKSPACE:
            self.typed = self.typed[:-1]
        elif key in (pg.K_RETURN, pg.K_KP_ENTER):
            if not repeat:  # a held Enter mustn't confirm something by accident
                self.enter()
        elif event.unicode and event.unicode.isprintable() and not ctrl:
            self.typed += event.unicode

    # --- Actions ----------------------------------------------------------------

    def click(self, pos: tuple[int, int], mods: int = 0) -> None:
        """A left click (not a drag) at a window position."""
        if self.frame is None:
            return
        button = self.frame.button_at(pos)
        if button is not None:
            self.press_button(button)
            return
        square = self.frame.square_at(pos)
        if square is None:
            return
        piece = self.place
        if mods & self.pg.KMOD_SHIFT:
            piece = PieceType.WALL
        elif mods & self.pg.KMOD_CTRL:
            piece = PieceType.CAP
        self.session.cancel_pending()
        outcome = self.builder.click(self.game, square, piece)
        if outcome.move is not None:
            move = outcome.move
            self.play(lambda: self.session.play(move))
        elif outcome.message:
            self.message = (BAD if outcome.error else INFO), outcome.message
        else:
            self.message = None

    def press_button(self, name: str) -> None:
        if name.startswith("place:"):
            self.place = PieceType(name.removeprefix("place:"))
        elif name == "nav:start":
            self.step(self.session.to_start)
        elif name == "nav:back":
            self.step(self.session.back)
        elif name == "nav:forward":
            self.step(self.session.forward)
        elif name == "nav:end":
            self.step(self.session.to_end)

    def enter(self) -> None:
        """Enter: play the typed move, or confirm a move waiting for confirmation."""
        text = self.typed.strip()
        self.typed = ""
        if text:
            self.builder.cancel()
            self.play(lambda: self.session.play_text(text))
        elif self.session.pending is not None:
            self.play(self.session.confirm)

    def play(self, action: Callable[[], Feedback | None]) -> None:
        """Run a session action that may play a move, and show its feedback."""
        before = self.game.ply
        self.message = action()
        if self.game.ply != before:
            self.place = PieceType.FLAT  # walls and capstones are one-offs
            self.builder.cancel()

    def step(self, action: Callable[[], object]) -> None:
        """Move through the game's timeline; any move being built is dropped."""
        self.builder.cancel()
        self.session.cancel_pending()
        action()
        self.message = None

    def cancel(self) -> None:
        """Esc or right-click: drop whatever is in progress."""
        lifted = self.builder.cancel()
        pending = self.session.cancel_pending()
        if lifted or pending or self.typed:
            self.message = None
        self.typed = ""

    def update(self, seconds: float, pressed: Any) -> None:
        """Per-frame work: held arrow keys turn (left/right) and tilt (up/down) the board."""
        pg = self.pg
        if pg.key.get_mods() & pg.KMOD_CTRL:
            return  # Ctrl+arrows step through the game instead
        turn = pressed[pg.K_LEFT] - pressed[pg.K_RIGHT]
        tilt = pressed[pg.K_UP] - pressed[pg.K_DOWN]
        if turn or tilt:
            self.camera = self.camera.orbit(turn * KEY_SPEED * seconds, tilt * KEY_SPEED * seconds)
            self.dirty = True

    def resize(self, size: tuple[int, int]) -> tuple[int, int]:
        """Fit the layout to a new window size; returns the size actually used.

        That is the requested size, or bigger if it was below the minimum.
        """
        self.layout = self.render.layout_for(self.game.size, size)
        self.dirty = True
        return self.layout.window_size

    # --- Drawing ----------------------------------------------------------------

    def draw(self, surface: Any) -> None:
        render = self.render
        if surface.get_size() != self.layout.window_size:
            self.resize(surface.get_size())
        game = self.game
        board, floating = self.builder.preview(game.board)

        highlights: dict[Square, tuple[int, int, int, int]] = {}
        last_move = None
        if game.history:
            played = game.history[-1]
            last_move = format_move(played.move, flattened=played.flattened)
            for square in touched_squares(played.move):
                highlights[square] = render.LAST_MOVE
        if self.builder.lift is not None:
            for square in self.builder.targets(game):
                highlights[square] = render.TARGET
            highlights[self.builder.lift.start] = render.LIFTED
        if self.hover is not None and self.hover not in highlights and not game.is_over:
            highlights[self.hover] = render.HOVER

        kind, text = self.message or (INFO, None)
        colors = {INFO: render.TEXT, GOOD: render.ACCENT, BAD: render.ERROR}
        self.frame = render.draw_game(
            surface,
            game,
            self.layout,
            self.fonts,
            camera=self.camera,
            message=text,
            message_color=colors[kind],
            input_text=None if game.is_over else self.typed,
            extras=render.SceneExtras(board=board, floating=floating, highlights=highlights),
            controls=render.Controls(
                place=self.place,
                ply=game.ply,
                total=self.session.total,
                last_move=last_move,
            ),
        )
        self.dirty = False


def run(session: GameSession, message: Feedback | None = None) -> None:
    """Open the window on ``session`` and run until it is closed."""
    import pygame

    pygame.init()
    pygame.key.set_repeat(KEY_REPEAT_DELAY, KEY_REPEAT_INTERVAL)
    controller = Controller(session, message=message)
    pygame.display.set_mode(controller.layout.window_size, pygame.RESIZABLE)
    title = f"Tak {session.game.size}\u00d7{session.game.size}"
    if session.source is not None:
        title += f" \u2013 {session.source.name}"
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()

    while controller.running:
        seconds = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                size = controller.resize(event.size)
                if size != tuple(event.size):  # too small: hold it at the minimum
                    pygame.display.set_mode(size, pygame.RESIZABLE)
            else:
                controller.handle(event)
        controller.update(seconds, pygame.key.get_pressed())
        if controller.dirty:
            controller.draw(pygame.display.get_surface())
            pygame.display.flip()

    pygame.quit()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tak.gui",
        description="Play Tak in a 3D window, or step through a saved game.",
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "-s",
        "--size",
        type=int,
        choices=sorted(BOARD_CONFIG),
        help=f"board size for a new game (default: {DEFAULT_SIZE})",
    )
    start.add_argument(
        "-l",
        "--load",
        type=Path,
        metavar="FILE",
        help="open a saved game (PTN) to step through it, or to play on from any point",
    )
    add_save_arguments(parser)
    args = parser.parse_args(argv)

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("The GUI needs pygame. Install it with: pip install -e '.[gui]'", file=sys.stderr)
        return 1

    save_dir = saves_dir_from(args)
    message: Feedback | None = None
    if args.load is not None:
        try:
            session = GameSession.load(args.load, save_dir)
        except (OSError, ValueError) as exc:  # unreadable, not PTN, or an illegal move
            print(f"Could not open {args.load}: {exc}", file=sys.stderr)
            return 1
        message = (
            INFO,
            f"Opened {describe_path(args.load)} ({session.total} moves). Step through it "
            "with the buttons below or Ctrl+Left/Right.",
        )
    else:
        session = GameSession(Game(args.size or DEFAULT_SIZE), save_dir)
    run(session, message)
    return 0
