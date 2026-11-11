"""The pygame render: layout, the board and labels, pieces, and reserves."""

from __future__ import annotations

from collections.abc import Callable

import pygame
import pytest

from tak.engine import BOARD_CONFIG, Color, Game, Piece, PieceType, Square
from tak.gui import render


def frame(game: Game) -> pygame.Surface:
    layout = render.layout_for(game.size)
    surface = pygame.Surface(layout.window_size)
    render.draw_game(surface, game, layout, render.Fonts.default(), input_text="a1")
    return surface


def colors_used(surface: pygame.Surface) -> set[tuple[int, int, int]]:
    width, height = surface.get_size()
    return {surface.get_at((x, y))[:3] for x in range(0, width, 4) for y in range(0, height, 4)}


class TestLayout:
    @pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
    def test_window_fits_board_labels_and_panel(self, size: int) -> None:
        layout = render.layout_for(size)
        width, height = layout.window_size
        assert layout.board_rect.width == layout.board_rect.height == size * layout.square
        assert layout.board_rect.left == layout.margin  # room for rank numbers
        assert layout.board_rect.bottom + layout.margin <= height  # room for file letters
        assert layout.panel_rect.left >= layout.board_rect.right
        assert layout.panel_rect.right <= width

    def test_a1_is_bottom_left_and_ranks_go_up(self) -> None:
        layout = render.layout_for(5)
        a1 = layout.square_rect(Square.from_name("a1"))
        a2 = layout.square_rect(Square.from_name("a2"))
        e1 = layout.square_rect(Square.from_name("e1"))
        assert a1.bottom == layout.board_rect.bottom and a1.left == layout.board_rect.left
        assert a2.bottom == a1.top  # rank 2 sits above rank 1
        assert e1.right == layout.board_rect.right and e1.left > a1.left


class TestDrawing:
    @pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
    def test_draws_every_size_without_error(self, size: int) -> None:
        assert colors_used(frame(Game(size)))  # and doesn't crash

    def test_empty_board_shows_both_square_colors(self) -> None:
        used = colors_used(frame(Game(5)))
        assert render.BOARD_LIGHT in used
        assert render.BOARD_DARK in used

    def test_pieces_of_both_colors_are_visible_and_distinct(
        self, started: Callable[..., Game]
    ) -> None:
        used = colors_used(frame(started("a1 e5 b2 d4")))
        assert render.WHITE_FILL in used
        assert render.BLACK_FILL in used

    def test_piece_types_draw_differently(self) -> None:
        side = 40
        surfaces = []
        for piece_type in PieceType:
            surface = pygame.Surface((side, side))
            surface.fill(render.BOARD_LIGHT)
            render.draw_piece(surface, Piece(Color.WHITE, piece_type), pygame.Rect(4, 4, 32, 32))
            surfaces.append(pygame.image.tobytes(surface, "RGB"))
        assert len(set(surfaces)) == len(PieceType)

    def test_tall_stack_stays_inside_its_square(self) -> None:
        game = Game(5)
        stack = tuple(Piece(Color.WHITE, PieceType.FLAT) for _ in range(9)) + (
            Piece(Color.BLACK, PieceType.CAP),
        )
        game.board.set_stack(Square.from_name("c3"), stack)
        frame(game)  # must not raise, even squeezed


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
