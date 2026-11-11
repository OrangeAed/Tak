"""Turn game state into pixels: a 3D board with stacks, labels, and reserves.

The 3D work (camera, meshes, shading, draw order, picking) is in
:mod:`tak.gui.scene`; this module composes a frame from it: the board seen
through the camera on the left, and a panel on the right with both players'
reserves, whose turn it is, messages, the piece chooser, the buttons for
stepping through the game, and the move input line. The window and event loop
live in :mod:`tak.gui.app`.

How the pieces look: White plays light maple and Black dark walnut. Flat
stones are low octagonal tiles, walls are the same tile standing on edge at 45
degrees, and capstones are tall bevelled drums. Stacks are built up layer by
layer, so their makeup can be read from the side; stacks of two or more also
get a small height tag floating above them.

Drawing returns a :class:`Frame`, which remembers where everything ended up so
the app can tell which square or button is under the mouse.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field

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
    pick_square,
    piece_faces,
    stack_faces,
    stack_height,
)

#: The scene is drawn this many times larger and scaled down, to smooth edges...
SUPERSAMPLE = 2
#: ...unless the viewport is bigger than this many pixels, when that gets slow.
SUPERSAMPLE_LIMIT = 1_000_000

#: How far carried stones float above the stack beneath them, in squares.
FLOAT_GAP = 0.55

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

# --- Palette -----------------------------------------------------------------------

BACKGROUND = (40, 36, 33)
TABLE_TOP = (58, 52, 46)  # the backdrop behind the board, lighter at the top
TABLE_BOTTOM = (30, 27, 25)
LABEL = (240, 226, 196)
TEXT = (236, 229, 214)
TEXT_DIM = (170, 161, 142)
ACCENT = (240, 196, 90)  # highlights: whose turn, Tak/Tinue calls, chosen piece
ERROR = (235, 120, 100)
PANEL_BG = (58, 53, 47)
PANEL_RULE = (92, 70, 44)
INPUT_BG = (36, 33, 30)
BUTTON_BG = (44, 40, 36)
TAG_BG = (28, 24, 21)
SHADOW = (0, 0, 0, 80)

#: Square highlights, drawn as translucent washes over the board.
HOVER: RGBA = (255, 250, 235, 55)
LAST_MOVE: RGBA = (250, 205, 90, 70)
LIFTED: RGBA = (110, 170, 255, 110)
TARGET: RGBA = (120, 225, 130, 85)


# --- Layout ------------------------------------------------------------------------

DEFAULT_WINDOW = (1080, 680)
MIN_WINDOW = (860, 600)
PANEL_WIDTH = 280
MARGIN = 16


@dataclass(frozen=True)
class Layout:
    """Where everything goes: the 3D viewport on the left, the panel on the right."""

    board_size: int
    viewport: pygame.Rect
    panel_rect: pygame.Rect
    window_size: tuple[int, int]


def layout_for(board_size: int, window_size: tuple[int, int] = DEFAULT_WINDOW) -> Layout:
    """Fit the viewport and panel to a window (no smaller than :data:`MIN_WINDOW`)."""
    width = max(window_size[0], MIN_WINDOW[0])
    height = max(window_size[1], MIN_WINDOW[1])
    view_width = width - PANEL_WIDTH - 3 * MARGIN
    viewport = pygame.Rect(MARGIN, MARGIN, view_width, height - 2 * MARGIN)
    panel = pygame.Rect(viewport.right + MARGIN, MARGIN, PANEL_WIDTH, viewport.height)
    return Layout(board_size, viewport, panel, (width, height))


# --- Fonts -------------------------------------------------------------------------


@dataclass(frozen=True)
class Fonts:
    text: pygame.font.Font  # panel text
    small: pygame.font.Font  # hints

    @classmethod
    def default(cls) -> Fonts:
        if not pygame.font.get_init():
            pygame.font.init()
        return cls(text=pygame.font.Font(None, 26), small=pygame.font.Font(None, 20))


@functools.lru_cache(maxsize=32)
def _font(size: int) -> pygame.font.Font:
    """A font at any pixel size, for labels that scale with the board."""
    return pygame.font.Font(None, max(8, size))


# --- What to draw besides the game itself ----------------------------------------


@dataclass(frozen=True)
class SceneExtras:
    """Things the app wants shown on the board that aren't part of the game state."""

    board: Board | None = None  # draw this instead of the game's board (a move preview)
    floating: tuple[Square, tuple[Piece, ...]] | None = None  # stones in the air, and where
    highlights: Mapping[Square, RGBA] = field(default_factory=dict)


@dataclass(frozen=True)
class Controls:
    """The state of the panel's controls."""

    place: PieceType = PieceType.FLAT  # what a click on an empty square places
    ply: int = 0  # moves played to reach the position shown
    total: int = 0  # moves in the game, counting any after the position shown
    last_move: str | None = None


@dataclass
class Frame:
    """Where a drawn frame put things, so clicks can be matched to them."""

    view: View
    supersample: int
    viewport: pygame.Rect
    board_size: int
    heights: dict[Square, float]  # how tall each stack stood, for picking
    buttons: dict[str, pygame.Rect] = field(default_factory=dict)

    def to_scene(self, pos: tuple[int, int]) -> tuple[float, float]:
        """Window pixel -> pixel in the (supersampled) scene."""
        return (
            (pos[0] - self.viewport.left) * self.supersample,
            (pos[1] - self.viewport.top) * self.supersample,
        )

    def to_window(self, point: tuple[float, float, float]) -> tuple[int, int]:
        """A point in the world -> window pixel."""
        x, y = self.view.project(point)
        return (
            self.viewport.left + round(x / self.supersample),
            self.viewport.top + round(y / self.supersample),
        )

    def square_at(self, pos: tuple[int, int]) -> Square | None:
        """The board square under a window pixel, if any."""
        if not self.viewport.collidepoint(pos):
            return None
        return pick_square(self.view, self.board_size, self.heights, *self.to_scene(pos))

    def button_at(self, pos: tuple[int, int]) -> str | None:
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                return name
        return None


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


def _draw_washes(surface: pygame.Surface, view: View, highlights: Mapping[Square, RGBA]) -> None:
    """Translucent color over highlighted squares."""
    if not highlights:
        return
    layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for square, color in highlights.items():
        f, r = square.file, square.rank
        corners = [(f, r), (f + 1, r), (f + 1, r + 1), (f, r + 1)]
        pygame.draw.polygon(layer, color, [view.project((x, y, 0.0)) for x, y in corners])
    surface.blit(layer, (0, 0))


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


def _draw_hover_shadow(
    surface: pygame.Surface, view: View, square: Square, piece: Piece, z: float
) -> None:
    """The shadow that carried stones cast on whatever is below them, so they look lifted."""
    center = (square.file + 0.5, square.rank + 0.5)
    outline = footprint(piece, center)
    points = [view.project((x, y, z)) for x, y in outline]
    left, top = min(p[0] for p in points), min(p[1] for p in points)
    right, bottom = max(p[0] for p in points), max(p[1] for p in points)
    layer = pygame.Surface((right - left + 2, bottom - top + 2), pygame.SRCALPHA)
    pygame.draw.polygon(layer, (0, 0, 0, 110), [(x - left, y - top) for x, y in points])
    surface.blit(layer, (left, top))


def _draw_tag(
    surface: pygame.Surface, view: View, square: Square, count: int, top: float, px: int, line: int
) -> None:
    """A small tag floating above a stack, saying how many pieces are in it."""
    x, y = view.project((square.file + 0.5, square.rank + 0.5, top + 0.08))
    text = _font(px).render(str(count), True, LABEL)
    box = text.get_rect().inflate(px // 2, px // 6)
    box.width = max(box.width, box.height)
    box.midbottom = (round(x), round(y))
    pygame.draw.rect(surface, TAG_BG, box, border_radius=box.height // 2)
    pygame.draw.rect(surface, PANEL_RULE, box, width=line, border_radius=box.height // 2)
    surface.blit(text, text.get_rect(center=box.center))


def draw_scene(
    surface: pygame.Surface,
    game: Game,
    layout: Layout,
    camera: Camera,
    extras: SceneExtras | None = None,
) -> Frame:
    """The board in 3D inside the viewport."""
    extras = extras or SceneExtras()
    board = extras.board or game.board
    viewport = layout.viewport
    ss = SUPERSAMPLE if viewport.width * viewport.height <= SUPERSAMPLE_LIMIT else 1
    scene = pygame.Surface((viewport.width * ss, viewport.height * ss))
    scene.blit(_backdrop(scene.get_size()), (0, 0))
    view = board_view(game.size, camera, scene.get_rect())

    paint(scene, view, _board_faces(game.size), line=ss)
    _draw_labels(scene, view, game.size)
    _draw_washes(scene, view, extras.highlights)
    _draw_shadows(scene, view, board)

    heights: dict[Square, float] = {}
    tag_px = max(12 * ss, round(_square_px(view, game.size) * 0.24))
    floating_square, floating = extras.floating or (None, ())
    # Tags are painted with their stacks, so a nearer stack hides a farther tag.
    for square in draw_order(game.size, view.eye):
        stack = board.stack(square)
        height = stack_height(stack)
        if stack:
            paint(scene, view, stack_faces(stack, square), line=ss)
        if len(stack) > 1 and not (square == floating_square and floating):
            _draw_tag(scene, view, square, len(stack), height, tag_px, ss)
        if square == floating_square and floating:
            _draw_hover_shadow(scene, view, square, floating[0], height)
            base = height + FLOAT_GAP
            paint(scene, view, stack_faces(floating, square, base), line=ss)
            height = base + stack_height(floating)
            _draw_tag(scene, view, square, len(floating), height, tag_px, ss)
        if height:
            heights[square] = height

    if ss == 1:
        surface.blit(scene, viewport)
    else:
        surface.blit(pygame.transform.smoothscale(scene, viewport.size), viewport)
    return Frame(view, ss, viewport.copy(), game.size, heights)


# --- Piece icons -------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def piece_icon(color: Color, piece_type: PieceType, size: int, pile: bool = True) -> pygame.Surface:
    """A small 3D picture of a piece. Stones show as a little pile unless ``pile`` is off."""
    canvas = pygame.Surface((size * SUPERSAMPLE, size * SUPERSAMPLE), pygame.SRCALPHA)
    piece = Piece(color, piece_type)
    if piece_type is PieceType.FLAT and pile:
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


# --- Panel -------------------------------------------------------------------------


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
    y += title.get_height() + 4

    icon = 36
    entries = [(PieceType.FLAT, reserve.stones)]
    if BOARD_CONFIG[game.size][1]:  # this board size has capstones
        entries.append((PieceType.CAP, reserve.caps))
    for column, (piece_type, count) in enumerate(entries):
        left = x + column * (width // 2)
        surface.blit(piece_icon(color, piece_type, icon), (left, y))
        label = fonts.text.render(f"\u00d7 {count}", True, TEXT)
        surface.blit(label, (left + icon + 6, y + (icon - label.get_height()) // 2))
    return y + icon + 10


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
    "Click: place a piece, or lift your stack",
    "Shift+click: wall \u00b7 Ctrl+click: capstone",
    "Right-click or Esc: cancel",
    "Drag or arrow keys: turn \u00b7 Wheel: zoom",
    "Tab: other side \u00b7 Home: reset view",
    "Ctrl+Left/Right or Ctrl+Z/Y: step",
    "Ctrl+Home/End: first/last move",
    "Ctrl+Q: quit",
)

#: The step buttons, left to right.
NAV_BUTTONS = ("nav:start", "nav:back", "nav:forward", "nav:end")


def _button(surface: pygame.Surface, rect: pygame.Rect, selected: bool, enabled: bool) -> None:
    pygame.draw.rect(surface, BUTTON_BG if enabled else PANEL_BG, rect, border_radius=6)
    border = ACCENT if selected else (PANEL_RULE if enabled else BUTTON_BG)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=6)


def _nav_glyph(surface: pygame.Surface, rect: pygame.Rect, name: str, color: RGB) -> None:
    """Step-button symbols drawn as shapes (the default font has no arrow glyphs)."""
    cx, cy = rect.center
    s = rect.height // 4
    left = name in ("nav:start", "nav:back")
    sign = -1 if left else 1
    tip = cx + sign * s
    pygame.draw.polygon(
        surface, color, [(tip, cy), (cx - sign * s, cy - s), (cx - sign * s, cy + s)]
    )
    if name in ("nav:start", "nav:end"):
        bar = tip + sign * 2
        pygame.draw.rect(
            surface, color, pygame.Rect(min(bar, bar + sign * 3), cy - s, 3, 2 * s + 1)
        )


def draw_panel(
    surface: pygame.Surface,
    game: Game,
    layout: Layout,
    fonts: Fonts,
    message: str | None = None,
    message_color: RGB = TEXT,
    input_text: str | None = None,
    controls: Controls | None = None,
) -> dict[str, pygame.Rect]:
    """The side panel. Returns the clickable buttons by name."""
    controls = controls or Controls(ply=game.ply, total=game.ply)
    buttons: dict[str, pygame.Rect] = {}
    panel = layout.panel_rect
    pygame.draw.rect(surface, PANEL_BG, panel, border_radius=8)
    x = panel.left + 16
    width = panel.width - 32

    # Built from the bottom up, so the top part gets whatever room is left.
    bottom = panel.bottom - 10
    hint_lines = [line for hint in HINTS for line in _wrap(hint, fonts.small, width)]
    bottom -= len(hint_lines) * 16
    for i, line in enumerate(hint_lines):
        surface.blit(fonts.small.render(line, True, TEXT_DIM), (x, bottom + i * 16))
    bottom -= 10

    if input_text is not None:
        box = pygame.Rect(x, bottom - 32, width, 32)
        pygame.draw.rect(surface, INPUT_BG, box, border_radius=6)
        prompt = fonts.text.render(f"> {input_text}_", True, TEXT)
        surface.blit(prompt, (box.left + 8, box.top + (box.height - prompt.get_height()) // 2))
        caption = fonts.small.render("Or type a move (PTN) and press Enter", True, TEXT_DIM)
        surface.blit(caption, (x, box.top - 18))
        bottom = box.top - 28

    # Step through the game: |<  <  >  >|   12 / 37
    nav = pygame.Rect(x, bottom - 30, 34, 30)
    for i, name in enumerate(NAV_BUTTONS):
        rect = nav.move(i * 40, 0)
        enabled = controls.ply > 0 if i < 2 else controls.ply < controls.total
        _button(surface, rect, False, enabled)
        _nav_glyph(surface, rect, name, TEXT if enabled else TEXT_DIM)
        buttons[name] = rect
    count = fonts.text.render(f"{controls.ply} / {controls.total}", True, TEXT)
    surface.blit(count, (nav.left + 4 * 40 + 6, nav.centery - count.get_height() // 2))
    bottom = nav.top - 10

    # What a click on an empty square places.
    if not game.is_over:
        owner = game.to_move.opponent if game.in_opening else game.to_move
        reserve = game.reserve(owner)
        choices = [PieceType.FLAT] if game.in_opening else [PieceType.FLAT, PieceType.WALL]
        if BOARD_CONFIG[game.size][1] and not game.in_opening:
            choices.append(PieceType.CAP)
        label = fonts.text.render("Place:", True, TEXT)
        row = pygame.Rect(x + label.get_width() + 10, bottom - 40, 40, 40)
        surface.blit(label, (x, row.centery - label.get_height() // 2))
        for i, piece_type in enumerate(choices):
            rect = row.move(i * 46, 0)
            available = reserve.caps if piece_type is PieceType.CAP else reserve.stones
            chosen = piece_type is controls.place or game.in_opening
            _button(surface, rect, chosen, available > 0)
            icon = piece_icon(owner, piece_type, 32, pile=False)
            surface.blit(icon, icon.get_rect(center=rect.center))
            buttons[f"place:{piece_type.value}"] = rect
        bottom = row.top - 8

    # Top part: reserves, status, and the latest message.
    y = panel.top + 12
    for color in Color:
        y = _reserve_row(surface, game, color, x, y, width, fonts)
    pygame.draw.line(surface, PANEL_RULE, (x, y), (panel.right - 16, y), 2)
    y += 10

    if game.result is not None:
        status = [describe_result(game, game.result)]
    else:
        status = [f"Turn {game.turn_number}: {game.to_move} to move"]
        if game.in_opening:
            status.append(f"placing a {game.to_move.opponent.value} flat")
    if controls.last_move:
        status.append(f"Last move: {controls.last_move}")
    lines = [(part, TEXT) for line in status for part in _wrap(line, fonts.text, width)]
    if message:
        lines += [(part, message_color) for part in _wrap(message, fonts.text, width)]
    for text, tint in lines:
        rendered = fonts.text.render(text, True, tint)
        if y + rendered.get_height() > bottom:
            break  # out of room: the controls below matter more
        surface.blit(rendered, (x, y))
        y += rendered.get_height() + 3
    return buttons


def _wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
    """Greedy word wrap by rendered pixel width.

    A word too wide for a line on its own (such as a file path) is broken up,
    after a ``/``, ``_`` or ``-`` where possible.
    """
    lines: list[str] = []
    current = ""
    words = [piece for word in text.split() for piece in _split_long(word, font, width)]
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and font.size(candidate)[0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _split_long(word: str, font: pygame.font.Font, width: int) -> list[str]:
    """``word`` cut into pieces that each fit in ``width`` pixels."""
    pieces = []
    while font.size(word)[0] > width and len(word) > 1:
        fit = 1
        while fit < len(word) and font.size(word[: fit + 1])[0] <= width:
            fit += 1
        breaks = [i + 1 for i in range(fit) if word[i] in "/_-"]
        cut = breaks[-1] if breaks else fit
        pieces.append(word[:cut])
        word = word[cut:]
    return [*pieces, word]


# --- Everything --------------------------------------------------------------------


def draw_game(
    surface: pygame.Surface,
    game: Game,
    layout: Layout,
    fonts: Fonts,
    camera: Camera | None = None,
    message: str | None = None,
    message_color: RGB = TEXT,
    input_text: str | None = None,
    extras: SceneExtras | None = None,
    controls: Controls | None = None,
) -> Frame:
    """One full frame: the 3D board and the side panel."""
    surface.fill(BACKGROUND)
    frame = draw_scene(surface, game, layout, camera or Camera(), extras)
    frame.buttons = draw_panel(
        surface, game, layout, fonts, message, message_color, input_text, controls
    )
    return frame
