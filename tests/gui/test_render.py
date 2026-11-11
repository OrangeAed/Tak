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


def frame(
    game: Game,
    camera: Camera | None = None,
    extras: render.SceneExtras | None = None,
    window: tuple[int, int] = render.DEFAULT_WINDOW,
) -> tuple[pygame.Surface, render.Frame]:
    layout = render.layout_for(game.size, window)
    surface = pygame.Surface(layout.window_size)
    drawn = render.draw_game(
        surface,
        game,
        layout,
        render.Fonts.default(),
        camera=camera,
        input_text="a1",
        extras=extras,
    )
    return surface, drawn


def color_on(
    game: Game,
    name: str,
    z: float,
    camera: Camera | None = None,
    extras: render.SceneExtras | None = None,
) -> tuple[int, ...]:
    """The pixel color where a point above the middle of a square is drawn."""
    surface, drawn = frame(game, camera, extras)
    square = Square.from_name(name)
    return tuple(surface.get_at(drawn.to_window((square.file + 0.5, square.rank + 0.5, z))))[:3]


def brightness(color: tuple[int, ...]) -> float:
    return sum(color) / 3


class TestLayout:
    @pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
    @pytest.mark.parametrize("window", [render.DEFAULT_WINDOW, (900, 620), (1920, 1080)])
    def test_viewport_and_panel_fit_the_window(self, size: int, window: tuple[int, int]) -> None:
        layout = render.layout_for(size, window)
        assert layout.window_size == window
        whole = pygame.Rect((0, 0), layout.window_size)
        assert whole.contains(layout.viewport)
        assert whole.contains(layout.panel_rect)
        assert not layout.viewport.colliderect(layout.panel_rect)

    def test_too_small_windows_get_the_minimum(self) -> None:
        assert render.layout_for(5, (100, 100)).window_size == render.MIN_WINDOW

    @pytest.mark.parametrize("window", [render.MIN_WINDOW, render.DEFAULT_WINDOW, (1920, 1080)])
    def test_panel_controls_do_not_overlap(self, window: tuple[int, int]) -> None:
        _, drawn = frame(Game(6), window=window)
        rects = list(drawn.buttons.values())
        for i, a in enumerate(rects):
            assert pygame.Rect((0, 0), window).contains(a)
            for b in rects[i + 1 :]:
                assert not a.colliderect(b)


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


class TestExtras:
    def test_carried_stones_float_above_the_board(self) -> None:
        game = Game(5)
        light = (WHITE_FLAT,) * 3
        extras = render.SceneExtras(floating=(Square.from_name("c3"), light))
        _, drawn = frame(game, extras=extras)
        assert drawn.heights[Square.from_name("c3")] > render.FLOAT_GAP  # lifted off the board
        # The stones are drawn up in the air, not on the board.
        z = render.FLOAT_GAP + 3 * FLAT_HEIGHT
        assert brightness(color_on(game, "c3", z, extras=extras)) > 170

    def test_highlights_tint_squares(self) -> None:
        game = Game(5)
        plain = color_on(game, "b2", 0.0)
        extras = render.SceneExtras(highlights={Square.from_name("b2"): render.TARGET})
        tinted = color_on(game, "b2", 0.0, extras=extras)
        assert tinted[1] > plain[1]  # greener

    def test_a_preview_board_is_drawn_instead_of_the_game_board(self) -> None:
        game = Game(5)
        preview = game.board.copy()
        preview.set_stack(Square.from_name("d4"), (BLACK_FLAT,))
        dark = color_on(game, "d4", FLAT_HEIGHT, extras=render.SceneExtras(board=preview))
        assert brightness(dark) < 110
        assert not game.board.stack(Square.from_name("d4"))


class TestFrame:
    def test_square_at_finds_squares_and_ignores_the_panel(self) -> None:
        _, drawn = frame(Game(5))
        for name in ("a1", "c3", "e5"):
            square = Square.from_name(name)
            pos = drawn.to_window((square.file + 0.5, square.rank + 0.5, 0.0))
            assert drawn.square_at(pos) == square
        assert drawn.square_at(drawn.buttons["nav:back"].center) is None

    def test_buttons(self) -> None:
        _, drawn = frame(Game(5))
        assert {"nav:start", "nav:back", "nav:forward", "nav:end", "place:F"} <= set(drawn.buttons)
        assert drawn.button_at(drawn.buttons["nav:end"].center) == "nav:end"

    def test_piece_choices_follow_the_game(self, started: Callable[..., Game]) -> None:
        _, drawn = frame(Game(5))
        assert {k for k in drawn.buttons if k.startswith("place:")} == {"place:F"}  # opening
        _, drawn = frame(started("a1 e5"))
        assert {k for k in drawn.buttons if k.startswith("place:")} == {
            "place:F",
            "place:S",
            "place:C",
        }
        _, drawn = frame(started("a1 c3", size=3))  # no capstones on 3x3
        assert "place:C" not in drawn.buttons
        _, drawn = frame(started("a1 c3 b1 b3 b2 a3 c2", size=3))  # game over
        assert not any(k.startswith("place:") for k in drawn.buttons)

    def test_big_windows_skip_supersampling(self) -> None:
        _, small = frame(Game(5))
        _, big = frame(Game(5), window=(1920, 1080))
        assert small.supersample == render.SUPERSAMPLE
        assert big.supersample == 1


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

    def test_wrap_breaks_a_long_path_at_separators(self) -> None:
        font = render.Fonts.default().text
        text = "Game saved to saves/2026-11-09_14-03-22_5x5_white-road.ptn"
        lines = render._wrap(text, font, 238)
        assert all(font.size(line)[0] <= 238 for line in lines)
        assert "".join(lines).replace(" ", "") == text.replace(" ", "")
        assert all(line[-1] in "/_-" for line in lines[1:-1])  # broke at separators
