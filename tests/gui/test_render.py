"""The rendered frame: layout, drawing, occlusion, wood colors, and the panel."""

from __future__ import annotations

import math
from collections.abc import Callable

import pygame
import pytest

from tak.engine import BOARD_CONFIG, Color, Game, Piece, PieceType, Square
from tak.gui import render
from tak.gui.scene import FLAT_HEIGHT, Camera

WHITE_FLAT = Piece(Color.WHITE, PieceType.FLAT)
BLACK_FLAT = Piece(Color.BLACK, PieceType.FLAT)


def frame(game: Game, camera: Camera | None = None) -> tuple[pygame.Surface, render.View]:
    layout = render.layout_for(game.size)
    surface = pygame.Surface(layout.window_size)
    view = render.draw_game(
        surface, game, layout, render.Fonts.default(), camera=camera, input_text="a1"
    )
    return surface, view


def color_on(game: Game, name: str, z: float, camera: Camera | None = None) -> tuple[int, ...]:
    """The pixel color where a point above the middle of a square is drawn."""
    surface, view = frame(game, camera)
    layout = render.layout_for(game.size)
    return tuple(surface.get_at(render.square_center(view, layout, Square.from_name(name), z)))[:3]


def brightness(color: tuple[int, ...]) -> float:
    return sum(color) / 3


class TestLayout:
    @pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
    def test_viewport_and_panel_fit_the_window(self, size: int) -> None:
        layout = render.layout_for(size)
        window = pygame.Rect((0, 0), layout.window_size)
        assert window.contains(layout.viewport)
        assert window.contains(layout.panel_rect)
        assert not layout.viewport.colliderect(layout.panel_rect)


class TestDrawing:
    @pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
    def test_draws_every_size_without_error(self, size: int) -> None:
        frame(Game(size))

    @pytest.mark.parametrize("yaw", [0.0, 1.0, 2.5, math.pi, 5.0])
    def test_draws_from_any_angle(self, yaw: float) -> None:
        game = Game(5)
        game.board.set_stack(Square.from_name("c3"), (WHITE_FLAT, BLACK_FLAT) * 6)
        frame(game, Camera(yaw=yaw, pitch=0.4))

    def test_light_pieces_are_light_wood_and_dark_pieces_dark_wood(self) -> None:
        game = Game(5)
        game.board.set_stack(Square.from_name("b2"), (WHITE_FLAT,))
        game.board.set_stack(Square.from_name("d4"), (BLACK_FLAT,))
        light = color_on(game, "b2", FLAT_HEIGHT)
        dark = color_on(game, "d4", FLAT_HEIGHT)
        assert brightness(light) > 170 > 110 > brightness(dark)
        for r, g, b in (light, dark):
            assert r > g > b  # wood, not black and white

    def test_pieces_stand_out_from_the_board(self) -> None:
        game = Game(5)
        board_color = color_on(game, "b2", 0.0)
        game.board.set_stack(Square.from_name("b2"), (WHITE_FLAT,))
        game.board.set_stack(Square.from_name("c2"), (BLACK_FLAT,))
        assert brightness(color_on(game, "b2", FLAT_HEIGHT)) > brightness(board_color) + 30
        assert brightness(color_on(game, "c2", FLAT_HEIGHT)) < brightness(board_color) - 40

    def test_a_tall_stack_in_front_hides_what_is_behind_it(self) -> None:
        # From White's side, c2 is in front of c3. A tall light stack on c2
        # must cover the dark flat on c3, however they are drawn.
        game = Game(5)
        game.board.set_stack(Square.from_name("c2"), (WHITE_FLAT,) * 25)
        game.board.set_stack(Square.from_name("c3"), (BLACK_FLAT,))
        # (The pixel falls on the stack's shaded side, so it is dimmer than a lit top,
        # but still far brighter than walnut ever gets.)
        assert brightness(color_on(game, "c3", FLAT_HEIGHT)) > 125
        # And from Black's side c3 is in front, so the dark flat is visible.
        assert brightness(color_on(game, "c3", FLAT_HEIGHT, Camera().flipped())) < 110

    def test_piece_types_look_different(self) -> None:
        images = set()
        for piece_type in PieceType:
            game = Game(5)
            game.board.set_stack(Square.from_name("c3"), (Piece(Color.WHITE, piece_type),))
            surface, _ = frame(game)
            images.add(pygame.image.tobytes(surface, "RGB"))
        assert len(images) == len(PieceType)

    @pytest.mark.parametrize("color", list(Color))
    @pytest.mark.parametrize("piece_type", [PieceType.FLAT, PieceType.CAP])
    def test_reserve_icons(self, color: Color, piece_type: PieceType) -> None:
        icon = render.piece_icon(color, piece_type, 40)
        assert icon.get_size() == (40, 40)
        assert icon.get_at((20, 20)).a > 0  # something is drawn in the middle


class TestPanel:
    def test_finished_game_renders_a_result(self, started: Callable[..., Game]) -> None:
        # c2 completes White's road from b1 up to c3.
        game = started("a1 c3 b1 b3 b2 a3 c2", size=3)
        assert game.result is not None
        assert "road" in render.describe_result(game, game.result)
        frame(game)  # result panel draws without error

    def test_wrap_breaks_long_messages(self) -> None:
        font = render.Fonts.default().text
        text = "a message long enough that it cannot possibly fit on one line"
        lines = render._wrap(text, font, 120)
        assert len(lines) > 1
        assert " ".join(lines) == text
