"""The 3D scene: camera, projection, meshes, and draw order."""

from __future__ import annotations

import math

import pygame
import pytest

from tak.engine import BOARD_CONFIG, Color, Piece, PieceType, Square
from tak.gui import scene
from tak.gui.scene import Camera

VIEWPORT = pygame.Rect(0, 0, 800, 600)


def sq(name: str) -> Square:
    return Square.from_name(name)


def ground(view: scene.View, name: str) -> tuple[float, float]:
    square = sq(name)
    return view.project((square.file + 0.5, square.rank + 0.5, 0.0))


class TestCamera:
    def test_pitch_is_kept_between_the_limits(self) -> None:
        assert Camera().orbit(0, 10).pitch == scene.MAX_PITCH
        assert Camera().orbit(0, -10).pitch == scene.MIN_PITCH

    def test_yaw_wraps_around(self) -> None:
        camera = Camera().orbit(math.tau + 0.5)
        assert camera.yaw == pytest.approx(0.5)

    def test_zoom_is_limited(self) -> None:
        assert Camera().zoomed(100).zoom == scene.MAX_ZOOM
        assert Camera().zoomed(0.01).zoom == scene.MIN_ZOOM

    def test_flipped_looks_from_the_other_side(self) -> None:
        assert Camera().flipped().yaw == pytest.approx(math.pi)
        assert Camera().flipped().flipped().yaw == pytest.approx(0.0)


class TestProjection:
    def test_whites_view_has_a1_at_front_left(self) -> None:
        view = scene.board_view(5, Camera(), VIEWPORT)
        a1, e1, a5 = ground(view, "a1"), ground(view, "e1"), ground(view, "a5")
        assert a1[0] < e1[0]  # files run left to right
        assert a1[1] > a5[1]  # rank 1 is nearer, so lower on screen

    def test_blacks_view_is_turned_around(self) -> None:
        view = scene.board_view(5, Camera().flipped(), VIEWPORT)
        a1, e1, a5 = ground(view, "a1"), ground(view, "e1"), ground(view, "a5")
        assert a1[0] > e1[0]
        assert a1[1] < a5[1]

    def test_higher_points_are_higher_on_screen(self) -> None:
        view = scene.board_view(5, Camera(), VIEWPORT)
        assert view.project((2.5, 2.5, 1.0))[1] < view.project((2.5, 2.5, 0.0))[1]

    @pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
    @pytest.mark.parametrize("yaw", [0.0, 0.7, math.pi, 4.0])
    def test_whole_board_fits_the_viewport(self, size: int, yaw: float) -> None:
        view = scene.board_view(size, Camera(yaw=yaw), VIEWPORT)
        for point in scene.fit_points(size):
            x, y = view.project(point)
            assert VIEWPORT.left <= x <= VIEWPORT.right
            assert VIEWPORT.top <= y <= VIEWPORT.bottom

    def test_zoom_magnifies(self) -> None:
        near = scene.board_view(5, Camera(zoom=2.0), VIEWPORT)
        far = scene.board_view(5, Camera(), VIEWPORT)
        width = lambda v: abs(ground(v, "e1")[0] - ground(v, "a1")[0])  # noqa: E731
        assert width(near) == pytest.approx(2 * width(far))


class TestMeshes:
    def test_box_faces_point_outward(self) -> None:
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        faces = scene.extrude(square, square, 0.0, 1.0, (0, 0, 0))
        normals = {tuple(round(c, 6) + 0.0 for c in face.normal) for face in faces}
        assert normals == {
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        }

    def test_pieces_stack_one_layer_at_a_time(self) -> None:
        flat = Piece(Color.WHITE, PieceType.FLAT)
        faces = scene.stack_faces((flat, flat, flat), sq("c3"))
        tops = sorted({round(p[2], 6) for face in faces for p in face.points})
        assert tops == pytest.approx(
            [0.0, scene.FLAT_HEIGHT, 2 * scene.FLAT_HEIGHT, 3 * scene.FLAT_HEIGHT]
        )

    def test_stack_height_depends_on_the_top_piece(self) -> None:
        flat = Piece(Color.WHITE, PieceType.FLAT)
        cap = Piece(Color.BLACK, PieceType.CAP)
        wall = Piece(Color.BLACK, PieceType.WALL)
        assert scene.stack_height(()) == 0
        assert scene.stack_height((flat,) * 4) == pytest.approx(4 * scene.FLAT_HEIGHT)
        assert scene.stack_height((flat, cap)) == pytest.approx(
            scene.FLAT_HEIGHT + scene.CAP_HEIGHT
        )
        assert scene.stack_height((wall,)) > scene.FLAT_HEIGHT  # a wall stands up

    def test_pieces_stay_inside_their_square(self) -> None:
        for piece_type in PieceType:
            faces = scene.stack_faces((Piece(Color.WHITE, piece_type),), sq("b2"))
            for face in faces:
                for x, y, _ in face.points:
                    assert 1 < x < 2 and 1 < y < 2

    def test_the_two_colors_are_light_and_dark_wood(self) -> None:
        light, dark = scene.PIECE_WOOD[Color.WHITE].base, scene.PIECE_WOOD[Color.BLACK].base
        assert sum(light) > 2 * sum(dark)
        for r, g, b in (light, dark):
            assert r > g > b  # warm brown tones, not grey


class TestDrawOrder:
    def test_every_square_once(self) -> None:
        view = scene.board_view(6, Camera(yaw=1.0), VIEWPORT)
        order = scene.draw_order(6, view.eye)
        assert sorted(order) == sorted(Square(f, r) for f in range(6) for r in range(6))

    def test_far_squares_first_from_whites_side(self) -> None:
        view = scene.board_view(5, Camera(), VIEWPORT)
        order = scene.draw_order(5, view.eye)
        assert order.index(sq("c5")) < order.index(sq("c3")) < order.index(sq("c1"))

    def test_far_squares_first_from_blacks_side(self) -> None:
        view = scene.board_view(5, Camera().flipped(), VIEWPORT)
        order = scene.draw_order(5, view.eye)
        assert order.index(sq("c1")) < order.index(sq("c3")) < order.index(sq("c5"))


class TestPicking:
    @pytest.mark.parametrize("yaw", [0.0, 1.2, math.pi, 4.5])
    def test_ground_point_undoes_the_projection(self, yaw: float) -> None:
        view = scene.board_view(5, Camera(yaw=yaw, pitch=0.7), VIEWPORT)
        for x, y in [(0.3, 0.2), (2.5, 2.5), (4.9, 1.1)]:
            back = view.ground_point(*view.project((x, y, 0.0)))
            assert back == pytest.approx((x, y))

    def test_ground_point_at_a_height(self) -> None:
        view = scene.board_view(5, Camera(), VIEWPORT)
        point = view.ground_point(*view.project((1.5, 3.5, 0.8)), z=0.8)
        assert point == pytest.approx((1.5, 3.5))

    def test_empty_board_picks_the_square_under_the_mouse(self) -> None:
        view = scene.board_view(5, Camera(), VIEWPORT)
        for name in ("a1", "c3", "e5", "b4"):
            square = sq(name)
            pixel = view.project((square.file + 0.5, square.rank + 0.5, 0.0))
            assert scene.pick_square(view, 5, {}, *pixel) == square

    def test_off_the_board_picks_nothing(self) -> None:
        view = scene.board_view(5, Camera(), VIEWPORT)
        rim = view.project((-0.3, 2.5, 0.0))
        assert scene.pick_square(view, 5, {}, *rim) is None
        assert scene.pick_square(view, 5, {}, 5.0, 5.0) is None  # above the horizon/backdrop

    def test_a_tall_stack_is_picked_where_it_covers_other_squares(self) -> None:
        view = scene.board_view(5, Camera(), VIEWPORT)
        side = view.project((2.5, 1.5, 1.5))  # up the side of a tall stack on c2...
        assert scene.pick_square(view, 5, {}, *side) != sq("c2")  # ...which, without it, is c3+
        assert scene.pick_square(view, 5, {sq("c2"): 2.0}, *side) == sq("c2")

    def test_the_nearer_of_two_stacks_wins(self) -> None:
        view = scene.board_view(5, Camera(), VIEWPORT)
        heights = {sq("c2"): 2.0, sq("c3"): 2.0}
        front_top = view.project((2.5, 1.5, 1.9))  # c2 stands in front of c3
        assert scene.pick_square(view, 5, heights, *front_top) == sq("c2")
