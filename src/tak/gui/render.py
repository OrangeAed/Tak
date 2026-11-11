"""Turn game state into pixels: the board, labels, stacks, and reserves.

Everything here draws onto a surface it is given and reads game state from the
engine; the window and event loop live in :mod:`tak.gui.app`.

How the pieces look:

- **Flat stones** are rounded square tiles.
- **Walls** (standing stones) are diagonal bars, a flat stone up on its edge.
- **Capstones** are circles with a ring.
- White pieces are light with a dark outline; black pieces are dark with a
  light outline.

Stacks are drawn bottom-up: every buried stone shows as a thin slab in its
owner's color, the top piece sits above them at full size, and stacks taller
than one also get a height badge in the corner of the square.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from tak.engine.board import BOARD_CONFIG, Board, Square
from tak.engine.game import Game
from tak.engine.pieces import Color, Piece, PieceType
from tak.engine.rules import GameResult, Reason, count_flats

# --- Palette -----------------------------------------------------------------

BACKGROUND = (46, 42, 38)
BOARD_LIGHT = (205, 175, 125)
BOARD_DARK = (166, 127, 80)
BOARD_EDGE = (92, 70, 44)
LABEL = (226, 216, 192)
TEXT = (236, 229, 214)
TEXT_DIM = (170, 161, 142)
ACCENT = (240, 196, 90)  # highlights: whose turn, Tak/Tinue calls
ERROR = (235, 120, 100)
PANEL_BG = (58, 53, 47)
INPUT_BG = (36, 33, 30)

WHITE_FILL = (242, 234, 216)
WHITE_EDGE = (90, 78, 58)
BLACK_FILL = (58, 54, 50)
BLACK_EDGE = (214, 205, 188)

_PIECE_COLORS: dict[Color, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    Color.WHITE: (WHITE_FILL, WHITE_EDGE),
    Color.BLACK: (BLACK_FILL, BLACK_EDGE),
}


# --- Layout ------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Where everything goes, for one board size and square size."""

    board_size: int
    square: int  # side of one board square, in pixels
    margin: int  # gutter around the board, where the labels live
    panel_width: int  # the reserve/status panel to the right of the board

    @property
    def board_px(self) -> int:
        return self.board_size * self.square

    @property
    def board_rect(self) -> pygame.Rect:
        return pygame.Rect(self.margin, self.margin, self.board_px, self.board_px)

    @property
    def panel_rect(self) -> pygame.Rect:
        x = self.margin + self.board_px + self.margin
        return pygame.Rect(x, self.margin, self.panel_width, self.board_px)

    @property
    def window_size(self) -> tuple[int, int]:
        width = self.margin + self.board_px + self.margin + self.panel_width + self.margin
        height = self.margin + self.board_px + self.margin
        return width, height

    def square_rect(self, square: Square) -> pygame.Rect:
        """The pixel rect of a board square; a1 is bottom-left, like the CLI."""
        x = self.margin + square.file * self.square
        y = self.margin + (self.board_size - 1 - square.rank) * self.square
        return pygame.Rect(x, y, self.square, self.square)


def layout_for(board_size: int, board_px: int = 560, panel_width: int = 250) -> Layout:
    """A layout whose board is roughly ``board_px`` pixels across."""
    square = max(24, board_px // board_size)
    margin = max(28, square // 2)
    return Layout(board_size, square, margin, panel_width)


# --- Fonts -------------------------------------------------------------------


@dataclass(frozen=True)
class Fonts:
    label: pygame.font.Font  # rank and file labels
    text: pygame.font.Font  # panel text
    small: pygame.font.Font  # stack-height badges, hints

    @classmethod
    def default(cls) -> Fonts:
        if not pygame.font.get_init():
            pygame.font.init()
        return cls(
            label=pygame.font.Font(None, 30),
            text=pygame.font.Font(None, 26),
            small=pygame.font.Font(None, 20),
        )


# --- Pieces ------------------------------------------------------------------


def draw_piece(surface: pygame.Surface, piece: Piece, rect: pygame.Rect) -> None:
    """Draw one piece inside ``rect``: tile, diagonal bar, or circled capstone."""
    fill, edge = _PIECE_COLORS[piece.color]
    line = max(2, rect.width // 16)
    if piece.type is PieceType.FLAT:
        radius = max(3, rect.width // 5)
        pygame.draw.rect(surface, fill, rect, border_radius=radius)
        pygame.draw.rect(surface, edge, rect, width=line, border_radius=radius)
    elif piece.type is PieceType.WALL:
        _draw_wall(surface, rect, fill, edge, line)
    else:  # capstone
        center = rect.center
        radius = rect.width // 2
        pygame.draw.circle(surface, fill, center, radius)
        pygame.draw.circle(surface, edge, center, radius, line)
        pygame.draw.circle(surface, edge, center, max(2, radius // 2), line)


def _draw_wall(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fill: tuple[int, int, int],
    edge: tuple[int, int, int],
    line: int,
) -> None:
    """A wall: a thick bar along the rect's diagonal, like a stone on its edge."""
    inset = rect.width // 6
    x0, y0 = rect.left + inset, rect.bottom - inset  # bottom-left
    x1, y1 = rect.right - inset, rect.top + inset  # top-right
    half = max(3.0, rect.width * 0.17)
    # The bar runs up-right, so the perpendicular offset is also diagonal.
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    px, py = -(y1 - y0) / length * half, (x1 - x0) / length * half
    points = [
        (x0 + px, y0 + py),
        (x1 + px, y1 + py),
        (x1 - px, y1 - py),
        (x0 - px, y0 - py),
    ]
    pygame.draw.polygon(surface, fill, points)
    pygame.draw.polygon(surface, edge, points, line)


def draw_stack(
    surface: pygame.Surface,
    stack: tuple[Piece, ...],
    square_rect: pygame.Rect,
    fonts: Fonts,
) -> None:
    """Draw a stack inside a board square, bottom piece first.

    Only the top piece can be a wall or capstone, so everything beneath it is
    drawn as a thin flat slab in its owner's color; the top piece gets the
    full-size glyph. Stacks taller than one get a height badge.
    """
    if not stack:
        return
    piece_side = int(square_rect.width * 0.60)
    slab_height = max(5, square_rect.width // 11)
    pad = max(3, square_rect.width // 18)

    below = stack[:-1]
    # Squeeze the slab spacing so even a tall stack stays inside the square,
    # and thin the slabs with it so they never overlap into a blob.
    room = square_rect.height - piece_side - 2 * pad
    step = min(slab_height + 2, max(3, room // len(below))) if below else 0
    slab_height = min(slab_height, max(3, step - 1))

    center_x = square_rect.centerx
    y = square_rect.bottom - pad  # bottom edge of the lowest slab
    for piece in below:
        slab = pygame.Rect(0, 0, piece_side, slab_height)
        slab.midbottom = (center_x, y)
        fill, edge = _PIECE_COLORS[piece.color]
        pygame.draw.rect(surface, fill, slab, border_radius=2)
        pygame.draw.rect(surface, edge, slab, width=1, border_radius=2)
        y -= step

    top = pygame.Rect(0, 0, piece_side, piece_side)
    top.midbottom = (center_x, y)
    draw_piece(surface, stack[-1], top)

    if len(stack) > 1:
        badge = fonts.small.render(str(len(stack)), True, LABEL)
        surface.blit(badge, badge.get_rect(topright=(square_rect.right - 4, square_rect.top + 3)))


# --- Board -------------------------------------------------------------------


def draw_board(surface: pygame.Surface, board: Board, layout: Layout, fonts: Fonts) -> None:
    """The checkered grid, the rank and file labels, and every stack."""
    pygame.draw.rect(surface, BOARD_EDGE, layout.board_rect.inflate(12, 12), border_radius=6)
    for square in board.squares():
        rect = layout.square_rect(square)
        color = BOARD_LIGHT if (square.file + square.rank) % 2 else BOARD_DARK
        pygame.draw.rect(surface, color, rect)
        draw_stack(surface, board.stack(square), rect, fonts)

    for file in range(board.size):  # file letters along the bottom
        label = fonts.label.render(chr(ord("a") + file), True, LABEL)
        x = layout.margin + file * layout.square + layout.square // 2
        y = layout.margin + layout.board_px + layout.margin // 2 + 4
        surface.blit(label, label.get_rect(center=(x, y)))
    for rank in range(board.size):  # rank numbers up the left side
        label = fonts.label.render(str(rank + 1), True, LABEL)
        x = layout.margin // 2 - 4
        y = layout.margin + (board.size - 1 - rank) * layout.square + layout.square // 2
        surface.blit(label, label.get_rect(center=(x, y)))


# --- Panel: reserves and status ----------------------------------------------


def _reserve_row(
    surface: pygame.Surface,
    game: Game,
    color: Color,
    x: int,
    y: int,
    width: int,
    fonts: Fonts,
) -> int:
    """One player's reserves; returns the y just below what was drawn."""
    reserve = game.reserve(color)
    moving = not game.is_over and game.to_move is color
    name = str(color) + ("  \u00b7 to move" if moving else "")
    title = fonts.text.render(name, True, ACCENT if moving else TEXT)
    surface.blit(title, (x, y))
    y += title.get_height() + 8

    icon = 26
    rect = pygame.Rect(x + 4, y, icon, icon)
    draw_piece(surface, Piece(color, PieceType.FLAT), rect)
    count = fonts.text.render(f"\u00d7 {reserve.stones}", True, TEXT)
    surface.blit(count, (rect.right + 10, y + (icon - count.get_height()) // 2))

    if BOARD_CONFIG[game.size][1]:  # this board size has capstones
        rect = pygame.Rect(x + width // 2, y, icon, icon)
        draw_piece(surface, Piece(color, PieceType.CAP), rect)
        count = fonts.text.render(f"\u00d7 {reserve.caps}", True, TEXT)
        surface.blit(count, (rect.right + 10, y + (icon - count.get_height()) // 2))
    return y + icon + 16


def describe_result(game: Game, result: GameResult) -> str:
    """A short sentence for the panel saying who won and how."""
    if result.winner is None:
        return f"Draw ({result.ptn})"
    if result.reason is Reason.ROAD:
        return f"{result.winner} wins with a road ({result.ptn})"
    white = count_flats(game.board, Color.WHITE)
    black = count_flats(game.board, Color.BLACK)
    return f"{result.winner} wins on flats, {white} to {black} ({result.ptn})"


def draw_panel(
    surface: pygame.Surface,
    game: Game,
    layout: Layout,
    fonts: Fonts,
    message: str | None = None,
    message_color: tuple[int, int, int] = TEXT,
    input_text: str | None = None,
) -> None:
    """Reserves for both players, whose turn it is, a message, and the input line."""
    panel = layout.panel_rect
    pygame.draw.rect(surface, PANEL_BG, panel, border_radius=8)
    x = panel.left + 16
    width = panel.width - 32
    y = panel.top + 14

    for color in Color:
        y = _reserve_row(surface, game, color, x, y, width, fonts)

    pygame.draw.line(surface, BOARD_EDGE, (x, y), (panel.right - 16, y), 2)
    y += 12

    if game.result is not None:
        status = describe_result(game, game.result)
    else:
        status = f"Turn {game.turn_number}: {game.to_move} to move"
        if game.in_opening:
            status += f"\nplacing a {game.to_move.opponent.value} flat"
    for line in status.split("\n"):
        rendered = fonts.text.render(line, True, TEXT)
        surface.blit(rendered, (x, y))
        y += rendered.get_height() + 4

    if message:
        y += 6
        for line in _wrap(message, fonts.text, width):
            rendered = fonts.text.render(line, True, message_color)
            surface.blit(rendered, (x, y))
            y += rendered.get_height() + 2

    if input_text is not None:
        box = pygame.Rect(x, panel.bottom - 76, width, 32)
        pygame.draw.rect(surface, INPUT_BG, box, border_radius=6)
        prompt = fonts.text.render(f"> {input_text}_", True, TEXT)
        surface.blit(prompt, (box.left + 8, box.top + (box.height - prompt.get_height()) // 2))
        hint = fonts.small.render("Type a PTN move, Enter to play", True, TEXT_DIM)
        surface.blit(hint, (x, box.bottom + 6))
        hint = fonts.small.render("Ctrl+Z undo \u00b7 Esc clear \u00b7 Ctrl+Q quit", True, TEXT_DIM)
        surface.blit(hint, (x, box.bottom + 22))


def _wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
    """Greedy word wrap by rendered pixel width."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.size(candidate)[0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# --- Everything --------------------------------------------------------------


def draw_game(
    surface: pygame.Surface,
    game: Game,
    layout: Layout,
    fonts: Fonts,
    message: str | None = None,
    message_color: tuple[int, int, int] = TEXT,
    input_text: str | None = None,
) -> None:
    """One full frame: background, board with labels, and the side panel."""
    surface.fill(BACKGROUND)
    draw_board(surface, game.board, layout, fonts)
    draw_panel(surface, game, layout, fonts, message, message_color, input_text)
