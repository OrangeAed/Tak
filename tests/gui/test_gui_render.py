"""Tests for drawing a game with pygame, checked through the pixels drawn."""

from __future__ import annotations

from collections.abc import Callable

import pygame
import pytest

from tak.engine import BOARD_CONFIG, Color, Game, GameResult, Piece, PieceType, Reason, Square
from tak.gui.render import (
    PIECE_FILL,
    compute_layout,
    describe_status,
    draw_game,
)

WINDOWS = [(1040, 720), (640, 420), (700, 1000), (1600, 500)]

W, B = Color.WHITE, Color.BLACK
FLAT, WALL, CAP = PieceType.FLAT, PieceType.WALL, PieceType.CAP


def drawn(game: Game, size: tuple[int, int] = (1040, 720)) -> tuple[pygame.Surface, pygame.Rect]:
    surface = pygame.Surface(size)
    layout = draw_game(surface, game)
    return surface, layout.board


def pixels_of(
    surface: pygame.Surface, color: tuple[int, int, int], area: pygame.Rect
) -> list[tuple[int, int]]:
    found = []
    for x in range(area.left, area.right):
        for y in range(area.top, area.bottom):
            if surface.get_at((x, y))[:3] == color:
                found.append((x, y))
    return found


def bounding_box(points: list[tuple[int, int]]) -> pygame.Rect:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return pygame.Rect(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)


def game_with(size: int, stacks: dict[str, tuple[Piece, ...]]) -> Game:
    game = Game(size)
    for name, stack in stacks.items():
        game.board.set_stack(Square.from_name(name), stack)
    return game


# --- Layout ---------------------------------------------------------------------


@pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
@pytest.mark.parametrize("window", WINDOWS)
def test_the_board_and_sidebar_fit_the_window_without_overlapping(
    size: int, window: tuple[int, int]
) -> None:
    layout = compute_layout(*window, size)
    screen = pygame.Rect(0, 0, *window)
    assert screen.contains(layout.board)
    assert screen.contains(layout.sidebar)
    assert not layout.board.colliderect(layout.sidebar)
    assert layout.board.width == layout.board.height == layout.cell * size


@pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
def test_a1_is_bottom_left_and_squares_tile_the_board(size: int) -> None:
    layout = compute_layout(1040, 720, size)
    top = size - 1
    assert layout.square_rect(Square(0, 0)).bottomleft == layout.board.bottomleft
    assert layout.square_rect(Square(top, top)).topright == layout.board.topright
    assert layout.square_rect(Square(1, 0)).left == layout.square_rect(Square(0, 0)).right
    assert layout.square_rect(Square(0, 1)).bottom == layout.square_rect(Square(0, 0)).top


# --- Pieces on the board ------------------------------------------------------------


def test_an_empty_board_has_no_pieces_on_it() -> None:
    surface, board = drawn(Game(5))
    for color in Color:
        assert not pixels_of(surface, PIECE_FILL[color], board)


def test_each_piece_is_drawn_in_its_owners_color_on_its_square() -> None:
    game = game_with(5, {"a1": (Piece(W, FLAT),), "e5": (Piece(B, FLAT),)})
    surface = pygame.Surface((1040, 720))
    layout = draw_game(surface, game)
    a1 = layout.square_rect(Square.from_name("a1"))
    e5 = layout.square_rect(Square.from_name("e5"))
    assert pixels_of(surface, PIECE_FILL[W], a1)
    assert not pixels_of(surface, PIECE_FILL[B], a1)
    assert pixels_of(surface, PIECE_FILL[B], e5)
    assert not pixels_of(surface, PIECE_FILL[W], e5)


@pytest.mark.parametrize("color", list(Color))
def test_flats_walls_and_capstones_have_different_shapes(color: Color) -> None:
    game = game_with(
        5, {"a1": (Piece(color, FLAT),), "b1": (Piece(color, WALL),), "c1": (Piece(color, CAP),)}
    )
    surface = pygame.Surface((1040, 720))
    layout = draw_game(surface, game)

    def shape(name: str) -> pygame.Rect:
        cell = layout.square_rect(Square.from_name(name))
        return bounding_box(pixels_of(surface, PIECE_FILL[color], cell))

    flat, wall, cap = shape("a1"), shape("b1"), shape("c1")
    assert flat.width > 3 * flat.height  # lying down
    assert wall.height > 1.5 * wall.width  # standing up
    assert abs(cap.width - cap.height) <= 2  # round
    assert wall.height > flat.height and cap.height > flat.height


def test_every_piece_of_a_short_stack_shows_with_the_top_piece_highest() -> None:
    stack = (Piece(W, FLAT), Piece(B, FLAT), Piece(W, FLAT), Piece(B, CAP))
    game = game_with(5, {"c3": stack})
    surface = pygame.Surface((1040, 720))
    layout = draw_game(surface, game)
    cell = layout.square_rect(Square.from_name("c3"))
    column = [surface.get_at((cell.centerx, y))[:3] for y in range(cell.bottom - 1, cell.top, -1)]
    # Walking up the middle of the square: the colors change piece by piece, bottom first.
    fills = [c for c in column if c in PIECE_FILL.values()]
    order = [c for i, c in enumerate(fills) if i == 0 or c != fills[i - 1]]
    assert order == [PIECE_FILL[W], PIECE_FILL[B], PIECE_FILL[W], PIECE_FILL[B]]


@pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
def test_a_very_tall_stack_stays_inside_its_square(size: int) -> None:
    stones = BOARD_CONFIG[size][0]
    stack = tuple(Piece(W if i % 2 else B, FLAT) for i in range(2 * stones)) + (Piece(W, WALL),)
    game = game_with(size, {"b2": stack})
    surface = pygame.Surface((1040, 720))
    layout = draw_game(surface, game)
    cell = layout.square_rect(Square.from_name("b2"))
    for color in Color:
        points = pixels_of(surface, PIECE_FILL[color], layout.board)
        assert points
        assert cell.contains(bounding_box(points))


# --- Reserves and status ------------------------------------------------------------


def test_reserves_show_each_players_own_pieces() -> None:
    surface = pygame.Surface((1040, 720))
    layout = draw_game(surface, Game(5))
    sidebar = layout.sidebar
    top_half = pygame.Rect(sidebar.left, sidebar.top, sidebar.width, sidebar.height // 2)
    bottom_half = pygame.Rect(sidebar.left, sidebar.centery, sidebar.width, sidebar.height // 2)
    assert pixels_of(surface, PIECE_FILL[B], top_half)  # Black's reserve is at the top
    assert not pixels_of(surface, PIECE_FILL[W], top_half)
    assert pixels_of(surface, PIECE_FILL[W], bottom_half)
    assert not pixels_of(surface, PIECE_FILL[B], bottom_half)


def test_reserve_shrinks_as_pieces_are_played(started: Callable[..., Game]) -> None:
    def reserve_area(game: Game, color: Color) -> int:
        surface = pygame.Surface((1040, 720))
        layout = draw_game(surface, game)
        return len(pixels_of(surface, PIECE_FILL[color], layout.sidebar))

    fresh = reserve_area(Game(5), Color.WHITE)
    assert reserve_area(started("a1 e5 b1", size=5), Color.WHITE) < fresh
    assert reserve_area(started("a1 e5 b1 e4 Ca2", size=5), Color.WHITE) < reserve_area(
        started("a1 e5 b1 e4 a2", size=5), Color.WHITE
    )


def test_an_empty_reserve_draws_no_pieces() -> None:
    game = Game(3)
    game.reserves[W] = type(game.reserves[W])(0, 0)
    surface = pygame.Surface((1040, 720))
    layout = draw_game(surface, game)
    assert not pixels_of(surface, PIECE_FILL[W], layout.sidebar)
    assert pixels_of(surface, PIECE_FILL[B], layout.sidebar)


def test_status_says_whose_turn_it_is_or_how_the_game_ended(
    started: Callable[..., Game],
) -> None:
    assert describe_status(Game(5)) == "Turn 1: White to move"
    assert describe_status(started("a1 e5 b1", size=5)) == "Turn 2: Black to move"
    won = started("a1 a2 b2 a3 c2", size=3)
    assert won.result == GameResult(W, Reason.ROAD)
    assert describe_status(won) == "White wins with a road  (R-0)"
    won.result = GameResult(B, Reason.FLAT)
    assert describe_status(won) == "Black wins on flats  (0-F)"
    won.result = GameResult(None, Reason.FLAT)
    assert describe_status(won) == "Draw  (1/2-1/2)"


@pytest.mark.parametrize("size", sorted(BOARD_CONFIG))
@pytest.mark.parametrize("window", WINDOWS)
def test_every_size_draws_in_every_window(size: int, window: tuple[int, int]) -> None:
    game = game_with(size, {"a1": (Piece(W, FLAT), Piece(B, WALL)), "b1": (Piece(B, CAP),)})
    surface, board = drawn(game, window)
    assert pixels_of(surface, PIECE_FILL[B], board)
