"""A small 3D renderer for the board, written against plain pygame.

pygame only draws 2D shapes, so this module does the 3D part itself: an orbit
camera with perspective projection, pieces and the board built as convex
polyhedra, flat shading from a light over the viewer's shoulder, and painting
back to front.

World units are board squares: square ``a1`` covers ``0..1`` on both ground
axes, ``x`` runs along the files (a, b, ...), ``y`` along the ranks (1, 2, ...),
and ``z`` points up, with the playing surface at ``z = 0``.

Why painting back to front is exact here: every stack stays inside its own
square's column, so any two stacks are separated by a vertical plane, and the
one on the camera's side of that plane can never be hidden by the other.
Sorting squares by their ground distance from the camera respects every such
plane, and inside a stack the camera is always above, so bottom-to-top is
right too. Within one convex piece only the faces turned toward the camera are
drawn, and those never overlap each other.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

import pygame

from tak.engine.board import Square
from tak.engine.pieces import Color, Piece, PieceType

Vec3 = tuple[float, float, float]
Point = tuple[float, float]
RGB = tuple[int, int, int]


# --- Vector helpers ------------------------------------------------------------


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(a: Vec3) -> Vec3:
    length = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / length, a[1] / length, a[2] / length)


def _shade(color: RGB, factor: float) -> RGB:
    r, g, b = (max(0, min(255, round(channel * factor))) for channel in color)
    return (r, g, b)


# --- Camera and projection --------------------------------------------------------

DEFAULT_PITCH = math.radians(50)
MIN_PITCH = math.radians(18)
MAX_PITCH = math.radians(89)
MIN_ZOOM, MAX_ZOOM = 0.5, 3.0


@dataclass(frozen=True)
class Camera:
    """Where the viewer stands, orbiting the middle of the board.

    ``yaw`` 0 looks from White's side (rank 1 nearest, file a on the left);
    ``pi`` looks from Black's. ``pitch`` is the angle above the table.
    """

    yaw: float = 0.0
    pitch: float = DEFAULT_PITCH
    zoom: float = 1.0

    def orbit(self, d_yaw: float, d_pitch: float = 0.0) -> Camera:
        pitch = min(MAX_PITCH, max(MIN_PITCH, self.pitch + d_pitch))
        return replace(self, yaw=(self.yaw + d_yaw) % math.tau, pitch=pitch)

    def zoomed(self, factor: float) -> Camera:
        return replace(self, zoom=min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor)))

    def flipped(self) -> Camera:
        """The same view from the other player's side of the table."""
        return self.orbit(math.pi)


class View:
    """A camera placed in the world and fitted to a viewport: 3D points -> pixels.

    The projection is scaled and centered so that every point in ``fit`` lands
    inside ``viewport`` (times ``margin``), then magnified by the camera's zoom.
    """

    def __init__(
        self,
        camera: Camera,
        target: Vec3,
        distance: float,
        viewport: pygame.Rect,
        fit: Iterable[Vec3],
        margin: float = 0.92,
    ) -> None:
        ground = distance * math.cos(camera.pitch)
        self.eye: Vec3 = (
            target[0] + ground * math.sin(camera.yaw),
            target[1] - ground * math.cos(camera.yaw),
            target[2] + distance * math.sin(camera.pitch),
        )
        self.forward = _normalize(_sub(target, self.eye))
        self.right = _normalize(_cross(self.forward, (0.0, 0.0, 1.0)))
        self.up = _cross(self.right, self.forward)

        # Light comes from above, over the viewer's left shoulder, so the lit
        # sides stay the same however the board is turned.
        level = _normalize((self.forward[0], self.forward[1], 0.0))
        self.light = _normalize(
            (
                -0.45 * self.right[0] - 0.55 * level[0],
                -0.45 * self.right[1] - 0.55 * level[1],
                1.0,
            )
        )

        self.scale, self.offset = 1.0, (0.0, 0.0)
        raw = [self.project(point) for point in fit]
        xs, ys = [p[0] for p in raw], [p[1] for p in raw]
        width, height = (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0
        self.scale = min(viewport.width / width, viewport.height / height) * margin * camera.zoom
        self.offset = (
            viewport.centerx - self.scale * (min(xs) + max(xs)) / 2,
            viewport.centery - self.scale * (min(ys) + max(ys)) / 2,
        )

    def project(self, point: Vec3) -> Point:
        d = _sub(point, self.eye)
        depth = _dot(d, self.forward)
        return (
            self.offset[0] + self.scale * _dot(d, self.right) / depth,
            self.offset[1] - self.scale * _dot(d, self.up) / depth,
        )

    def ground_point(self, x: float, y: float, z: float = 0.0) -> Point | None:
        """The point at height ``z`` that shows at pixel ``(x, y)``: projection undone.

        Returns ``None`` if that pixel looks above the horizon.
        """
        a = (x - self.offset[0]) / self.scale
        b = -(y - self.offset[1]) / self.scale
        ray = tuple(
            f + a * r + b * u for f, r, u in zip(self.forward, self.right, self.up, strict=True)
        )
        if ray[2] >= 0:
            return None
        t = (z - self.eye[2]) / ray[2]
        return (self.eye[0] + t * ray[0], self.eye[1] + t * ray[1])

    def faces_camera(self, normal: Vec3, point: Vec3) -> bool:
        return _dot(normal, _sub(self.eye, point)) > 0

    def brightness(self, normal: Vec3) -> float:
        return 0.52 + 0.58 * max(0.0, _dot(normal, self.light))


# --- Faces and painting -----------------------------------------------------------


@dataclass
class Face:
    """A flat convex polygon, wound counter-clockwise as seen from outside."""

    points: Sequence[Vec3]
    color: RGB
    grain: Sequence[Sequence[Vec3]] = ()  # wood-grain polylines lying on the face
    grain_color: RGB | None = None
    edge: bool = True  # outline the face (off for the facets of round surfaces)
    normal: Vec3 = field(init=False)
    center: Vec3 = field(init=False)

    def __post_init__(self) -> None:
        # Newell's method: robust for any planar polygon.
        nx = ny = nz = 0.0
        count = len(self.points)
        for i, (x1, y1, z1) in enumerate(self.points):
            x2, y2, z2 = self.points[(i + 1) % count]
            nx += (y1 - y2) * (z1 + z2)
            ny += (z1 - z2) * (x1 + x2)
            nz += (x1 - x2) * (y1 + y2)
        self.normal = _normalize((nx, ny, nz))
        self.center = (
            sum(p[0] for p in self.points) / count,
            sum(p[1] for p in self.points) / count,
            sum(p[2] for p in self.points) / count,
        )


def paint(surface: pygame.Surface, view: View, faces: Iterable[Face], line: int = 2) -> None:
    """Draw the camera-facing faces, in the order given, with shading and grain."""
    for face in faces:
        if not view.faces_camera(face.normal, face.center):
            continue
        light = view.brightness(face.normal)
        fill = _shade(face.color, light)
        points = [view.project(p) for p in face.points]
        pygame.draw.polygon(surface, fill, points)
        if face.grain and face.grain_color:
            grain = _shade(face.grain_color, light)
            for polyline in face.grain:
                pygame.draw.lines(surface, grain, False, [view.project(p) for p in polyline], line)
        if face.edge:
            pygame.draw.polygon(surface, _shade(fill, 0.62), points, line)


def extrude(
    bottom: Sequence[Point],
    top: Sequence[Point],
    z0: float,
    z1: float,
    color: RGB,
    smooth: bool = False,
) -> list[Face]:
    """A convex solid from a footprint (counter-clockwise) up to a top outline.

    ``top`` has the same number of corners as ``bottom``; different sizes make
    a tapered solid. The bottom face is left out, since the camera never looks
    up from under the table. ``smooth`` leaves the side facets unoutlined, for
    round shapes. The top face is last.
    """
    faces = []
    count = len(bottom)
    for i in range(count):
        j = (i + 1) % count
        corners = [(*bottom[i], z0), (*bottom[j], z0), (*top[j], z1), (*top[i], z1)]
        faces.append(Face(corners, color, edge=not smooth))
    faces.append(Face([(*p, z1) for p in top], color))
    return faces


# --- Wood --------------------------------------------------------------------------


@dataclass(frozen=True)
class Wood:
    base: RGB
    grain: RGB


#: White plays light maple, Black plays dark walnut.
LIGHT_WOOD = Wood(base=(234, 203, 152), grain=(204, 168, 112))
DARK_WOOD = Wood(base=(92, 56, 34), grain=(66, 38, 22))
PIECE_WOOD = {Color.WHITE: LIGHT_WOOD, Color.BLACK: DARK_WOOD}

#: The board: a reddish frame with two tones of square.
FRAME_WOOD = Wood(base=(112, 62, 38), grain=(94, 50, 30))
SQUARE_WOODS = (
    Wood(base=(186, 144, 99), grain=(170, 128, 85)),
    Wood(base=(150, 107, 68), grain=(136, 95, 58)),
)


def _jitter(color: RGB, rng: random.Random, amount: float = 0.05) -> RGB:
    """A slightly different tint, so no two pieces look stamped from one mold."""
    return _shade(color, rng.uniform(1 - amount, 1 + amount))


def _grain_lines(
    outline: Sequence[Point],
    z: float,
    rng: random.Random,
    count: int,
    angle: float = 0.0,
    wobble: float = 0.008,
    inset: float = 0.03,
) -> list[list[Vec3]]:
    """Gently wavy lines across a convex outline, at height ``z``: wood grain."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    local = [(x * cos_a + y * sin_a, -x * sin_a + y * cos_a) for x, y in outline]
    v_lo, v_hi = min(v for _, v in local), max(v for _, v in local)
    lines = []
    for k in range(count):
        v = v_lo + (v_hi - v_lo) * (k + 0.5 + rng.uniform(-0.3, 0.3)) / count
        crossings = []
        for (u1, v1), (u2, v2) in zip(local, local[1:] + local[:1], strict=True):
            if (v1 - v) * (v2 - v) < 0:
                crossings.append(u1 + (v - v1) * (u2 - u1) / (v2 - v1))
        if len(crossings) < 2:
            continue
        u_lo, u_hi = min(crossings) + inset, max(crossings) - inset
        if u_hi <= u_lo:
            continue
        phase, frequency = rng.uniform(0, math.tau), rng.uniform(6, 14)
        polyline = []
        for i in range(9):
            u = u_lo + (u_hi - u_lo) * i / 8
            w = v + wobble * math.sin(frequency * u + phase)
            polyline.append((u * cos_a - w * sin_a, u * sin_a + w * cos_a, z))
        lines.append(polyline)
    return lines


# --- Pieces ------------------------------------------------------------------------

FLAT_WIDTH = 0.62  # across a flat stone, in squares
FLAT_HEIGHT = 0.12  # one layer of a stack
FLAT_CHAMFER = 0.09
WALL_THICKNESS = 0.17
CAP_RADIUS = 0.25
CAP_HEIGHT = 0.50
_CAP_SIDES = 28


def _octagon(cx: float, cy: float, width: float, chamfer: float) -> list[Point]:
    h = width / 2
    c = h - chamfer
    corners = [(-c, -h), (c, -h), (h, -c), (h, c), (c, h), (-c, h), (-h, c), (-h, -c)]
    return [(cx + x, cy + y) for x, y in corners]


def _circle(cx: float, cy: float, radius: float, sides: int = _CAP_SIDES) -> list[Point]:
    return [
        (cx + radius * math.cos(math.tau * i / sides), cy + radius * math.sin(math.tau * i / sides))
        for i in range(sides)
    ]


def _wall_outline(cx: float, cy: float) -> list[Point]:
    """A standing stone's footprint: a thin rectangle turned 45 degrees."""
    h, t = FLAT_WIDTH / 2, WALL_THICKNESS / 2
    cos_a = sin_a = math.sqrt(0.5)
    corners = [(-h, -t), (h, -t), (h, t), (-h, t)]
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]


def footprint(piece: Piece, center: Point) -> list[Point]:
    """The outline a piece covers on whatever it stands on."""
    if piece.type is PieceType.FLAT:
        return _octagon(*center, FLAT_WIDTH, FLAT_CHAMFER)
    if piece.type is PieceType.WALL:
        return _wall_outline(*center)
    return _circle(*center, CAP_RADIUS)


def piece_faces(piece: Piece, center: Point, z: float, seed: int) -> list[Face]:
    """The faces of one piece standing on height ``z`` at ``center``."""
    rng = random.Random(seed)
    wood = PIECE_WOOD[piece.color]
    color = _jitter(wood.base, rng)
    cx, cy = center

    if piece.type is PieceType.FLAT:
        outline = _octagon(cx, cy, FLAT_WIDTH, FLAT_CHAMFER)
        faces = extrude(outline, outline, z, z + FLAT_HEIGHT, color)
        top = faces[-1]
        top.grain = _grain_lines(outline, z + FLAT_HEIGHT, rng, 4, rng.choice((0, math.pi / 2)))
        top.grain_color = wood.grain
        return faces

    if piece.type is PieceType.WALL:
        outline = _wall_outline(cx, cy)
        faces = extrude(outline, outline, z, z + FLAT_WIDTH, color)
        for side in (faces[0], faces[2]):  # the two broad faces get grain
            side.grain = _wall_grain(side.points, rng)
            side.grain_color = wood.grain
        return faces

    # Capstone: a slightly tapered drum with a bevelled top and end-grain rings.
    body_top = CAP_HEIGHT - 0.07
    base, waist, crown = CAP_RADIUS, CAP_RADIUS * 0.9, CAP_RADIUS * 0.62
    faces = extrude(_circle(cx, cy, base), _circle(cx, cy, waist), z, z + body_top, color, True)
    faces.pop()  # the body's top is hidden under the bevel
    bevel = extrude(
        _circle(cx, cy, waist), _circle(cx, cy, crown), z + body_top, z + CAP_HEIGHT, color, True
    )
    top = bevel[-1]
    ox, oy = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)
    rings = []
    for radius in (0.035, 0.07, 0.105):
        ring = _circle(cx + ox, cy + oy, radius, 18)
        rings.append([(x, y, z + CAP_HEIGHT) for x, y in ring + ring[:1]])  # closed loop
    top.grain = rings
    top.grain_color = wood.grain
    return faces + bevel


def _wall_grain(corners: Sequence[Vec3], rng: random.Random) -> list[list[Vec3]]:
    """Horizontal grain across a wall's broad face (bottom edge, then top edge)."""
    x0, y0, z0 = corners[0]
    x1, y1, _ = corners[1]
    z1 = corners[3][2]
    lines = []
    for k in range(5):
        height = z0 + (z1 - z0) * (k + 0.5 + rng.uniform(-0.3, 0.3)) / 5
        phase = rng.uniform(0, math.tau)
        polyline = []
        for i in range(9):
            t = 0.06 + 0.88 * i / 8
            wobble = 0.008 * math.sin(10 * t + phase)
            polyline.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, height + wobble))
        lines.append(polyline)
    return lines


def stack_faces(stack: Sequence[Piece], square: Square, base: float = 0.0) -> list[Face]:
    """Every piece in a stack, bottom first, each resting on the one below.

    ``base`` lifts the whole stack off the board, for stones being carried.
    """
    center = (square.file + 0.5, square.rank + 0.5)
    faces: list[Face] = []
    for level, piece in enumerate(stack):
        seed = (square.file * 8 + square.rank) * 256 + level * 2 + (piece.color is Color.BLACK)
        faces.extend(piece_faces(piece, center, base + level * FLAT_HEIGHT, seed))
    return faces


def stack_height(stack: Sequence[Piece]) -> float:
    """How tall a stack stands, in squares."""
    if not stack:
        return 0.0
    top = stack[-1].type
    crown = {PieceType.FLAT: FLAT_HEIGHT, PieceType.WALL: FLAT_WIDTH, PieceType.CAP: CAP_HEIGHT}
    return (len(stack) - 1) * FLAT_HEIGHT + crown[top]


# --- Board ---------------------------------------------------------------------------

BOARD_RIM = 0.55  # width of the frame around the squares
BOARD_DEPTH = 0.28


def board_faces(size: int) -> list[Face]:
    """The board: a framed slab with wooden squares set into its top."""
    rng = random.Random(size)
    lo, hi = -BOARD_RIM, size + BOARD_RIM
    outline = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    faces = extrude(outline, outline, -BOARD_DEPTH, 0.0, FRAME_WOOD.base)
    frame_top = faces[-1]
    frame_top.grain = _grain_lines(outline, 0.0, rng, size * 4, wobble=0.02, inset=0.0)
    frame_top.grain_color = FRAME_WOOD.grain

    for square in (Square(f, r) for r in range(size) for f in range(size)):
        wood = SQUARE_WOODS[(square.file + square.rank) % 2]
        f, r = square.file, square.rank
        outline = [(f, r), (f + 1.0, r), (f + 1.0, r + 1.0), (f, r + 1.0)]
        angle = 0.0 if (f + r) % 2 else math.pi / 2  # parquet: alternate the grain
        faces.append(
            Face(
                [(x, y, 0.0) for x, y in outline],
                _jitter(wood.base, rng, 0.02),
                _grain_lines(outline, 0.0, rng, 3, angle, wobble=0.015),
                wood.grain,
            )
        )
    return faces


def fit_points(size: int, headroom: float = 0.8) -> list[Vec3]:
    """Points the view must show: the whole board, plus room above it for stacks."""
    lo, hi = -BOARD_RIM, size + BOARD_RIM
    corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    points = [(x, y, z) for x, y in corners for z in (0.0, -BOARD_DEPTH)]
    points += [(x, y, headroom) for x in (0.0, float(size)) for y in (0.0, float(size))]
    return points


def board_view(size: int, camera: Camera, viewport: pygame.Rect) -> View:
    """The camera looking at a ``size`` board, fitted to ``viewport``."""
    center = (size / 2, size / 2, 0.0)
    return View(
        camera, center, 2.2 * size + 3, viewport=viewport, fit=fit_points(size), margin=0.96
    )


def draw_order(size: int, eye: Vec3) -> list[Square]:
    """Every square, farthest from the camera first (see the module docstring)."""
    squares = [Square(f, r) for r in range(size) for f in range(size)]
    return sorted(
        squares,
        key=lambda s: -math.hypot(s.file + 0.5 - eye[0], s.rank + 0.5 - eye[1]),
    )


# --- Picking: which square is under the mouse ---------------------------------------


def pick_square(
    view: View, size: int, heights: dict[Square, float], x: float, y: float
) -> Square | None:
    """The square shown at pixel ``(x, y)``, counting the stacks standing on squares.

    ``heights`` gives how tall the stack on each square stands. A click on the
    side or top of a stack picks that stack's square, even where it stands in
    front of other squares; anywhere else, the square on the board below the
    mouse is picked. Returns ``None`` off the board.
    """
    half = FLAT_WIDTH / 2
    for square in reversed(draw_order(size, view.eye)):  # nearest first: drawn on top
        height = heights.get(square, 0.0)
        if height <= 0:
            continue
        cx, cy = square.file + 0.5, square.rank + 0.5
        corners = [
            view.project((cx + dx, cy + dy, z))
            for dx in (-half, half)
            for dy in (-half, half)
            for z in (0.0, height)
        ]
        if _inside(_convex_hull(corners), (x, y)):
            return square
    point = view.ground_point(x, y)
    if point is None:
        return None
    file, rank = math.floor(point[0]), math.floor(point[1])
    if 0 <= file < size and 0 <= rank < size:
        return Square(file, rank)
    return None


def _convex_hull(points: list[Point]) -> list[Point]:
    """The convex hull, counter-clockwise (Andrew's monotone chain)."""
    ordered = sorted(set(points))
    if len(ordered) < 3:
        return ordered

    def turn(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in ordered:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(ordered):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _inside(hull: list[Point], point: Point) -> bool:
    """True if ``point`` is inside (or on) a convex polygon given counter-clockwise."""
    if len(hull) < 3:
        return False
    for (x1, y1), (x2, y2) in zip(hull, hull[1:] + hull[:1], strict=True):
        if (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1) < 0:
            return False
    return True
