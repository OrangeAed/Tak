"""Draw a game of Tak onto a pygame surface.

Everything here is drawing only: no window, no event loop, no input. The
layout is worked out from the surface size, so the same code draws any board
size into any window.

Stacks are drawn from the side, bottom piece lowest, so every piece in a
stack can be seen:

- a flat stone is a wide, thin slab,
- a standing stone (wall) is a narrow, upright slab,
- a capstone is a disc.

White pieces are ivory and Black pieces are charcoal. Each player's reserve
is drawn in the sidebar as piles of ten stones, plus any capstones.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from tak.engine.board import BOARD_CONFIG, Square
from tak.engine.game import Game
from tak.engine.pieces import Color, Piece, PieceType
from tak.engine.rules import GameResult, Reason

RGB = tuple[int, int, int]

BACKGROUND: RGB = (38, 41, 48)
BOARD_EDGE: RGB = (92, 62, 36)
SQUARE_LIGHT: RGB = (214, 180, 132)
SQUARE_DARK: RGB = (200, 164, 115)
GRID_LINE: RGB = (120, 86, 52)
LABEL: RGB = (226, 226, 232)
MUTED: RGB = (150, 154, 165)
PANEL: RGB = (52, 56, 66)
HIGHLIGHT: RGB = (240, 190, 70)

#: Fill and outline for each player's pieces. The outlines separate pieces of
#: the same color stacked on top of each other.
PIECE_FILL: dict[Color, RGB] = {Color.WHITE: (244, 238, 222), Color.BLACK: (44, 44, 50)}
PIECE_EDGE: dict[Color, RGB] = {Color.WHITE: (96, 82, 66), Color.BLACK: (150, 150, 162)}

#: Stones per pile in the reserve display.
PILE = 10

# Piece proportions, as fractions of a board square.
_PAD = 0.1  # space between the pieces and the square's edge
_SLAB_WIDTH = 0.7
_SLAB_MAX = 0.18  # a flat's thickness when there's plenty of room
_SLAB_MIN = 0.08  # the thinnest a flat gets before lower pieces are left out
_WALL_WIDTH = 0.2
_WALL_HEIGHT = 0.38
_CAP_SIZE = 0.34


@dataclass(frozen=True)
class Layout:
    """Where everything goes on a surface of a given size."""

    board: pygame.Rect  # the squares only, without the labels
    cell: int  # the side of one square, in pixels
    size: int  # squares per side
    sidebar: pygame.Rect  # the reserves and the status line

    def square_rect(self, square: Square) -> pygame.Rect:
        """The pixel rectangle of ``square``. Rank 1 is at the bottom, file a on the left."""
        x = self.board.left + square.file * self.cell
        y = self.board.top + (self.size - 1 - square.rank) * self.cell
        return pygame.Rect(x, y, self.cell, self.cell)


def compute_layout(width: int, height: int, size: int) -> Layout:
    """Fit a ``size`` x ``size`` board, its labels, and the sidebar into ``width`` x ``height``."""
    margin = max(12, min(width, height) // 30)
    label_band = max(20, min(width, height) // 22)  # room for rank and file labels
    sidebar_width = max(220, min(360, width * 3 // 10))

    room_across = width - sidebar_width - 3 * margin - label_band
    room_down = height - 2 * margin - label_band
    cell = max(8, min(room_across, room_down) // size)
    side = cell * size

    # Center the board (plus its labels) in the space left of the sidebar.
    left_area = width - sidebar_width - margin
    board_left = max(margin + label_band, (left_area - side + label_band) // 2)
    board_top = max(margin, (height - side - label_band) // 2)
    board = pygame.Rect(board_left, board_top, side, side)

    # The sidebar matches the board's height (but isn't cramped when the board
    # is small), so a tall window doesn't leave it mostly empty.
    sidebar_height = min(height - 2 * margin, max(side + label_band, 320))
    sidebar_top = max(margin, board_top + (side + label_band - sidebar_height) // 2)
    sidebar = pygame.Rect(
        width - sidebar_width - margin, sidebar_top, sidebar_width, max(0, sidebar_height)
    )
    return Layout(board, cell, size, sidebar)


def _font(px: int, bold: bool = False) -> pygame.font.Font:
    # Not cached: a Font outlives pygame.quit() but crashes if used after it,
    # and making one is cheap (well under a millisecond).
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, max(8, px))  # pygame's bundled font, same everywhere
    font.set_bold(bold)
    return font


def _blit_text(
    surface: pygame.Surface,
    text: str,
    px: int,
    color: RGB,
    bold: bool = False,
    **position: tuple[int, int] | int,
) -> pygame.Rect:
    """Draw ``text``, placed with pygame.Rect keywords such as ``center=(x, y)``."""
    image = _font(px, bold).render(text, True, color)
    rect = image.get_rect(**position)
    surface.blit(image, rect)
    return rect


# --- Pieces ---------------------------------------------------------------------


def draw_slab(surface: pygame.Surface, color: Color, rect: pygame.Rect) -> None:
    """A flat stone (or a wall, if ``rect`` is tall and narrow), seen from the side."""
    radius = max(1, min(rect.width, rect.height) // 4)
    edge = 1 if rect.height < 8 else 2
    pygame.draw.rect(surface, PIECE_FILL[color], rect, border_radius=radius)
    pygame.draw.rect(surface, PIECE_EDGE[color], rect, width=edge, border_radius=radius)


def draw_cap(surface: pygame.Surface, color: Color, center: tuple[int, int], diameter: int) -> None:
    """A capstone: a disc with a ring inside so it reads as a cap even when small."""
    radius = max(2, diameter // 2)
    pygame.draw.circle(surface, PIECE_FILL[color], center, radius)
    pygame.draw.circle(surface, PIECE_EDGE[color], center, radius, width=max(1, radius // 8))
    if radius >= 6:
        pygame.draw.circle(surface, PIECE_EDGE[color], center, radius // 2, width=1)


def draw_stack(surface: pygame.Surface, stack: tuple[Piece, ...], cell: pygame.Rect) -> None:
    """Draw ``stack`` (bottom piece first) inside the board square ``cell``."""
    if not stack:
        return
    size = cell.width
    pad = round(size * _PAD)
    bottom = cell.bottom - pad
    top_piece = stack[-1]
    flats = stack if top_piece.type is PieceType.FLAT else stack[:-1]

    # Room for the flats: whatever the wall or capstone on top doesn't need.
    standing = {PieceType.WALL: _WALL_HEIGHT, PieceType.CAP: _CAP_SIZE}.get(top_piece.type, 0.0)
    room = (size - 2 * pad) - round(size * standing)
    thickness = 0
    if flats:
        thickness = max(3, min(round(size * _SLAB_MAX), room // len(flats)))
        thickness = max(thickness, round(size * _SLAB_MIN))
        # If the stack is too tall to show in full, show its top pieces; the
        # height badge still gives the full count.
        shown = max(1, min(len(flats), room // thickness))
        flats = flats[-shown:]

    slab_width = round(size * _SLAB_WIDTH)
    y = bottom
    for piece in flats:
        y -= thickness
        draw_slab(
            surface,
            piece.color,
            pygame.Rect(cell.centerx - slab_width // 2, y, slab_width, thickness),
        )

    if top_piece.type is PieceType.WALL:
        width, height = round(size * _WALL_WIDTH), round(size * _WALL_HEIGHT)
        draw_slab(
            surface,
            top_piece.color,
            pygame.Rect(cell.centerx - width // 2, y - height, width, height),
        )
    elif top_piece.type is PieceType.CAP:
        diameter = round(size * _CAP_SIZE)
        draw_cap(surface, top_piece.color, (cell.centerx, y - diameter // 2), diameter)

    if len(stack) > 1:
        _draw_height_badge(surface, len(stack), cell)


def _draw_height_badge(surface: pygame.Surface, height: int, cell: pygame.Rect) -> None:
    """The number of pieces in a stack, in the square's top-left corner."""
    px = max(12, cell.width // 4)
    image = _font(px, bold=True).render(str(height), True, LABEL)
    inset = max(2, cell.width // 20)
    badge = image.get_rect().inflate(px // 2, 2)
    badge.topleft = (cell.left + inset, cell.top + inset)
    pygame.draw.rect(surface, BACKGROUND, badge, border_radius=badge.height // 2)
    surface.blit(image, image.get_rect(center=badge.center))


# --- Board ----------------------------------------------------------------------


def draw_board(surface: pygame.Surface, game: Game, layout: Layout) -> None:
    """The squares, the pieces on them, and the rank and file labels."""
    board = layout.board
    frame = max(3, min(10, layout.cell // 12))
    pygame.draw.rect(surface, BOARD_EDGE, board.inflate(2 * frame, 2 * frame), border_radius=frame)

    for square in game.board.squares():
        rect = layout.square_rect(square)
        shade = SQUARE_LIGHT if (square.file + square.rank) % 2 else SQUARE_DARK
        pygame.draw.rect(surface, shade, rect)
        pygame.draw.rect(surface, GRID_LINE, rect, width=1)
        draw_stack(surface, game.board.stack(square), rect)

    px = max(14, min(layout.cell // 3, 34))
    gap = frame + px // 2 + 2
    for index in range(layout.size):
        file_x = board.left + index * layout.cell + layout.cell // 2
        rank_y = board.bottom - index * layout.cell - layout.cell // 2
        _blit_text(surface, chr(ord("a") + index), px, LABEL, center=(file_x, board.bottom + gap))
        _blit_text(surface, str(index + 1), px, LABEL, center=(board.left - gap, rank_y))


# --- Sidebar ----------------------------------------------------------------------


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def draw_reserve(surface: pygame.Surface, game: Game, color: Color, area: pygame.Rect) -> None:
    """One player's unplayed pieces: piles of stones and any capstones, with counts."""
    reserve = game.reserve(color)
    max_stones, max_caps = BOARD_CONFIG[game.size]
    to_move = game.result is None and game.to_move is color

    pygame.draw.rect(surface, PANEL, area, border_radius=10)
    if to_move:
        pygame.draw.rect(surface, HIGHLIGHT, area, width=3, border_radius=10)

    inner = area.inflate(-24, -20)
    heading = max(18, min(30, area.width // 10))
    title = _blit_text(surface, str(color), heading, LABEL, bold=True, topleft=inner.topleft)
    if to_move:
        _blit_text(
            surface,
            "to move",
            heading * 3 // 4,
            HIGHLIGHT,
            midleft=(title.right + 10, title.centery),
        )

    counts = _plural(reserve.stones, "stone")
    if max_caps:
        counts += ", " + _plural(reserve.caps, "capstone")
    summary = _blit_text(
        surface, counts, heading * 3 // 4, MUTED, topleft=(inner.left, title.bottom + 6)
    )

    # The pieces themselves: every pile, then every capstone, side by side.
    pieces_area = pygame.Rect(
        inner.left, summary.bottom + 10, inner.width, inner.bottom - summary.bottom - 10
    )
    if pieces_area.height < 16:
        return
    # A tray in the board's color, so both players' pieces stand out against it.
    pygame.draw.rect(surface, SQUARE_DARK, pieces_area, border_radius=6)
    pieces_area.inflate_ip(-12, -12)
    piles = -(-max_stones // PILE)  # enough slots for a full reserve, so nothing moves
    # At most four slots across, so a small reserve doesn't stretch to fill the tray.
    slot = pieces_area.width // max(4, piles + max_caps)
    thickness = max(2, min(slot // 3, pieces_area.height // PILE))
    slab_width = slot * 17 // 20
    for index in range(piles):
        in_pile = min(PILE, reserve.stones - index * PILE)
        x = pieces_area.left + index * slot + (slot - slab_width) // 2
        for level in range(in_pile):
            y = pieces_area.bottom - (level + 1) * thickness
            draw_slab(surface, color, pygame.Rect(x, y, slab_width, thickness))
    diameter = min(slot * 3 // 5, pieces_area.height)
    for index in range(reserve.caps):
        center_x = pieces_area.left + (piles + index) * slot + slot // 2
        draw_cap(surface, color, (center_x, pieces_area.bottom - diameter // 2), diameter)


def describe_status(game: Game) -> str:
    """A short line: whose turn it is, or how the game ended."""
    if game.result is not None:
        return f"{_describe_result(game.result)}  ({game.result.ptn})"
    return f"Turn {game.turn_number}: {game.to_move} to move"


def _describe_result(result: GameResult) -> str:
    if result.winner is None:
        return "Draw"
    how = "with a road" if result.reason is Reason.ROAD else "on flats"
    return f"{result.winner} wins {how}"


def draw_sidebar(surface: pygame.Surface, game: Game, layout: Layout) -> None:
    """Black's reserve at the top, White's at the bottom (nearest rank 1), status between."""
    sidebar = layout.sidebar
    status_px = max(18, min(28, sidebar.width // 12))
    status_height = status_px * 2
    panel_height = (sidebar.height - status_height) // 2
    black = pygame.Rect(sidebar.left, sidebar.top, sidebar.width, panel_height)
    white = pygame.Rect(sidebar.left, sidebar.bottom - panel_height, sidebar.width, panel_height)
    draw_reserve(surface, game, Color.BLACK, black)
    draw_reserve(surface, game, Color.WHITE, white)
    color = HIGHLIGHT if game.result is not None else LABEL
    _blit_text(surface, describe_status(game), status_px, color, bold=True, center=sidebar.center)


def draw_game(surface: pygame.Surface, game: Game) -> Layout:
    """Draw the whole game onto ``surface``, sized to fit it. Returns the layout used."""
    layout = compute_layout(surface.get_width(), surface.get_height(), game.size)
    surface.fill(BACKGROUND)
    draw_board(surface, game, layout)
    draw_sidebar(surface, game, layout)
    return layout
