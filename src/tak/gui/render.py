"""Turn game state into pixels: a 3D board with stacks, labels, and reserves.

The 3D work (camera, meshes, shading, draw order) is in :mod:`tak.gui.scene`;
this module composes a frame from it: the board seen through the camera on the
left, and a panel on the right with both players' reserves, whose turn it is,
messages, and the move input line. The window and event loop live in
:mod:`tak.gui.app`.

How the pieces look: White plays light maple and Black dark walnut. Flat
stones are low octagonal tiles, walls are the same tile standing on edge at 45
degrees, and capstones are tall bevelled drums. Stacks are built up layer by
layer, so their makeup can be read from the side; stacks of two or more also
get a small height tag floating above them.

The scene is drawn at twice the window's resolution and scaled down, which
smooths the edges that pygame's polygon drawing would otherwise leave jagged.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import pygame

from tak.engine.board import BOARD_CONFIG, Board, Square
from tak.engine.game import Game
from tak.engine.pieces import Color, Piece, PieceType
from tak.engine.rules import GameResult, Reason, count_flats
from tak.gui.scene import (
    BOARD_RIM,
    Camera,
    View,
    board_faces,
    board_view,
    draw_order,
    footprint,
    paint,
    piece_faces,
    stack_faces,
    stack_height,
)

SUPERSAMPLE = 2

# --- Palette -----------------------------------------------------------------------

BACKGROUND = (40, 36, 33)
TABLE_TOP = (58, 52, 46)  # the backdrop behind the board, lighter at the top
TABLE_BOTTOM = (30, 27, 25)
LABEL = (240, 226, 196)
TEXT = (236, 229, 214)
TEXT_DIM = (170, 161, 142)
ACCENT = (240, 196, 90)  # highlights: whose turn, Tak/Tinue calls
ERROR = (235, 120, 100)
PANEL_BG = (58, 53, 47)
PANEL_RULE = (92, 70, 44)
INPUT_BG = (36, 33, 30)
TAG_BG = (28, 24, 21)
SHADOW = (0, 0, 0, 80)


# --- Layout ------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Where everything goes: the 3D viewport on the left, the panel on the right."""

    board_size: int
    viewport: pygame.Rect
    panel_rect: pygame.Rect
    window_size: tuple[int, int]


def layout_for(
    board_size: int,
    view_size: tuple[int, int] = (740, 620),
    panel_width: int = 270,
    margin: int = 16,
) -> Layout:
    # Wider than tall: a board seen at an angle is foreshortened front to back.
    viewport = pygame.Rect((margin, margin), view_size)
    panel = pygame.Rect(viewport.right + margin, margin, panel_width, viewport.height)
    return Layout(board_size, viewport, panel, (panel.right + margin, viewport.bottom + margin))


# --- Fonts -------------------------------------------------------------------------


@dataclass(frozen=True)
class Fonts:
    text: pygame.font.Font  # panel text
    small: pygame.font.Font  # hints, stack tags

    @classmethod
    def default(cls) -> Fonts:
        if not pygame.font.get_init():
            pygame.font.init()
        return cls(text=pygame.font.Font(None, 26), small=pygame.font.Font(None, 20))


@functools.lru_cache(maxsize=32)
def _font(size: int) -> pygame.font.Font:
    """A font at any pixel size, for labels that scale with the board."""
    return pygame.font.Font(None, max(8, size))


# --- The 3D scene ------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _backdrop(size: tuple[int, int]) -> pygame.Surface:
    """A vertical gradient behind the board, like a table under a lamp."""
    width, height = size
    surface = pygame.Surface(size)
    for y in range(height):
        t = y / max(1, height - 1)
        color = tuple(round(a + (b - a) * t) for a, b in zip(TABLE_TOP, TABLE_BOTTOM, strict=True))
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface


_board_faces = functools.lru_cache(maxsize=8)(board_faces)


def _square_px(view: View, size: int) -> float:
    """About how many pixels one square spans in the middle of the board."""
    x0, y0 = view.project((size / 2, size / 2, 0.0))
    x1, y1 = view.project((size / 2 + 1, size / 2, 0.0))
    x2, y2 = view.project((size / 2, size / 2 + 1, 0.0))
    return max(abs(x1 - x0) + abs(y1 - y0), abs(x2 - x0) + abs(y2 - y0))


def _draw_labels(surface: pygame.Surface, view: View, size: int) -> None:
    """Files (a, b, ...) and ranks (1, 2, ...) on all four sides of the frame."""
    font = _font(round(_square_px(view, size) * 0.34))
    middle = BOARD_RIM / 2
    for i in range(size):
        letter = font.render(chr(ord("a") + i), True, LABEL)
        number = font.render(str(i + 1), True, LABEL)
        for point in ((i + 0.5, -middle, 0.0), (i + 0.5, size + middle, 0.0)):
            surface.blit(letter, letter.get_rect(center=view.project(point)))
        for point in ((-middle, i + 0.5, 0.0), (size + middle, i + 0.5, 0.0)):
            surface.blit(number, number.get_rect(center=view.project(point)))


def _draw_shadows(surface: pygame.Surface, view: View, board: Board) -> None:
    """Soft contact shadows under every stack, cast away from the light."""
    layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    dx, dy = -view.light[0] * 0.06, -view.light[1] * 0.06
    for square in board.squares():
        stack = board.stack(square)
        if not stack:
            continue
        center = (square.file + 0.5 + dx, square.rank + 0.5 + dy)
        outline = footprint(stack[0], center)
        # Grow the outline a touch so the shadow peeks out around the piece.
        grown = [
            (center[0] + (x - center[0]) * 1.12, center[1] + (y - center[1]) * 1.12)
            for x, y in outline
        ]
        pygame.draw.polygon(layer, SHADOW, [view.project((x, y, 0.0)) for x, y in grown])
    surface.blit(layer, (0, 0))


def _draw_tag(
    surface: pygame.Surface, view: View, square: Square, stack: tuple[Piece, ...], px: int
) -> None:
    """A small tag floating above a stack, saying how many pieces are in it."""
    x, y = view.project((square.file + 0.5, square.rank + 0.5, stack_height(stack) + 0.08))
    text = _font(px).render(str(len(stack)), True, LABEL)
    box = text.get_rect().inflate(px // 2, px // 6)
    box.width = max(box.width, box.height)
    box.midbottom = (round(x), round(y))
    pygame.draw.rect(surface, TAG_BG, box, border_radius=box.height // 2)
    pygame.draw.rect(surface, PANEL_RULE, box, width=SUPERSAMPLE, border_radius=box.height // 2)
    surface.blit(text, text.get_rect(center=box.center))


def draw_scene(surface: pygame.Surface, game: Game, layout: Layout, camera: Camera) -> View:
    """The board in 3D inside the viewport. Returns the (supersampled) view used."""
    viewport = layout.viewport
    scene = pygame.Surface((viewport.width * SUPERSAMPLE, viewport.height * SUPERSAMPLE))
    scene.blit(_backdrop(scene.get_size()), (0, 0))
    view = board_view(game.size, camera, scene.get_rect())

    paint(scene, view, _board_faces(game.size), line=SUPERSAMPLE)
    _draw_labels(scene, view, game.size)
    _draw_shadows(scene, view, game.board)
    # Tags are painted with their stacks, so a nearer stack hides a farther tag.
    tag_px = max(12 * SUPERSAMPLE, round(_square_px(view, game.size) * 0.24))
    for square in draw_order(game.size, view.eye):
        stack = game.board.stack(square)
        if stack:
            paint(scene, view, stack_faces(stack, square), line=SUPERSAMPLE)
        if len(stack) > 1:
            _draw_tag(scene, view, square, stack, tag_px)

    surface.blit(pygame.transform.smoothscale(scene, viewport.size), viewport)
    return view


def square_center(view: View, layout: Layout, square: Square, z: float = 0.0) -> tuple[int, int]:
    """Where a point above the middle of ``square`` appears in the window."""
    x, y = view.project((square.file + 0.5, square.rank + 0.5, z))
    return (
        layout.viewport.left + round(x / SUPERSAMPLE),
        layout.viewport.top + round(y / SUPERSAMPLE),
    )


# --- Reserve icons -----------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def piece_icon(color: Color, piece_type: PieceType, size: int) -> pygame.Surface:
    """A small 3D picture of a piece, for the reserves. Stones show as a little pile."""
    canvas = pygame.Surface((size * SUPERSAMPLE, size * SUPERSAMPLE), pygame.SRCALPHA)
    piece = Piece(color, piece_type)
    if piece_type is PieceType.FLAT:
        faces = [f for level in range(3) for f in piece_faces(piece, (0, 0), level * 0.12, level)]
    else:
        faces = piece_faces(piece, (0.0, 0.0), 0.0, seed=1)
    points = [p for face in faces for p in face.points]
    view = View(
        Camera(yaw=-0.5, pitch=0.62),
        target=(0.0, 0.0, 0.2),
        distance=4.0,
        viewport=canvas.get_rect(),
        fit=points,
        margin=0.94,
    )
    paint(canvas, view, faces, line=SUPERSAMPLE)
    return pygame.transform.smoothscale(canvas, (size, size))


# --- Panel: reserves and status ----------------------------------------------------


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
    wood = "maple" if color is Color.WHITE else "walnut"
    title = fonts.text.render(
        f"{color} ({wood})" + ("  \u00b7 to move" if moving else ""),
        True,
        ACCENT if moving else TEXT,
    )
    surface.blit(title, (x, y))
    y += title.get_height() + 6

    icon = 40
    entries = [(PieceType.FLAT, reserve.stones)]
    if BOARD_CONFIG[game.size][1]:  # this board size has capstones
        entries.append((PieceType.CAP, reserve.caps))
    for column, (piece_type, count) in enumerate(entries):
        left = x + column * (width // 2)
        surface.blit(piece_icon(color, piece_type, icon), (left, y))
        label = fonts.text.render(f"\u00d7 {count}", True, TEXT)
        surface.blit(label, (left + icon + 6, y + (icon - label.get_height()) // 2))
    return y + icon + 14


def describe_result(game: Game, result: GameResult) -> str:
    """A short sentence for the panel saying who won and how."""
    if result.winner is None:
        return f"Draw ({result.ptn})"
    if result.reason is Reason.ROAD:
        return f"{result.winner} wins with a road ({result.ptn})"
    white = count_flats(game.board, Color.WHITE)
    black = count_flats(game.board, Color.BLACK)
    return f"{result.winner} wins on flats, {white} to {black} ({result.ptn})"


HINTS = (
    "Drag or arrow keys: turn the board",
    "Mouse wheel: zoom",
    "Tab: other side \u00b7 Home: reset view",
    "Ctrl+Z undo \u00b7 Esc clear \u00b7 Ctrl+Q quit",
)


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

    pygame.draw.line(surface, PANEL_RULE, (x, y), (panel.right - 16, y), 2)
    y += 12

    if game.result is not None:
        status = describe_result(game, game.result)
    else:
        status = f"Turn {game.turn_number}: {game.to_move} to move"
        if game.in_opening:
            status += f"\nplacing a {game.to_move.opponent.value} flat"
    for line in status.split("\n"):
        for part in _wrap(line, fonts.text, width):
            rendered = fonts.text.render(part, True, TEXT)
            surface.blit(rendered, (x, y))
            y += rendered.get_height() + 4

    if message:
        y += 6
        for line in _wrap(message, fonts.text, width):
            rendered = fonts.text.render(line, True, message_color)
            surface.blit(rendered, (x, y))
            y += rendered.get_height() + 2

    hint_y = panel.bottom - 12 - len(HINTS) * 16
    for i, hint in enumerate(HINTS):
        surface.blit(fonts.small.render(hint, True, TEXT_DIM), (x, hint_y + i * 16))

    if input_text is not None:
        box = pygame.Rect(x, hint_y - 44, width, 32)
        pygame.draw.rect(surface, INPUT_BG, box, border_radius=6)
        prompt = fonts.text.render(f"> {input_text}_", True, TEXT)
        surface.blit(prompt, (box.left + 8, box.top + (box.height - prompt.get_height()) // 2))
        caption = fonts.small.render("Type a PTN move, Enter to play", True, TEXT_DIM)
        surface.blit(caption, (x, box.top - 20))


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


# --- Everything --------------------------------------------------------------------


def draw_game(
    surface: pygame.Surface,
    game: Game,
    layout: Layout,
    fonts: Fonts,
    camera: Camera | None = None,
    message: str | None = None,
    message_color: tuple[int, int, int] = TEXT,
    input_text: str | None = None,
) -> View:
    """One full frame: the 3D board and the side panel. Returns the board's view."""
    surface.fill(BACKGROUND)
    view = draw_scene(surface, game, layout, camera or Camera())
    draw_panel(surface, game, layout, fonts, message, message_color, input_text)
    return view
