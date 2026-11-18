"""The rendered frame: layout, drawing, occlusion, wood colors, and the panel."""

from __future__ import annotations

import math
from collections.abc import Callable

import pygame
import pytest

from tak.ai.assess import Assessment
from tak.ai.explain import Explanation
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


EXPLANATION = Explanation(
    "Why Learner played 3b2-",
    "Of the 105 moves that don't hand you a win, this is the one its learned evaluation "
    "liked best.",
    (
        "It takes the lead on flats, 4 to 2.",
        "It blocks your road (you are now about 4 squares from one, up from 3) and builds "
        "toward its own road (about 2 squares to go, down from 4).",
        "It takes ground near the center, where a road can head in any direction.",
    ),
    "How it rates its position now: ahead (+0.25, on a scale of -1 to +1).",
)


class TestExplanation:
    def draw(
        self,
        window: tuple[int, int] = render.DEFAULT_WINDOW,
        explained: tuple[Square, ...] = (),
        controls: render.Controls | None = None,
    ) -> render.Frame:
        game = Game(5)
        layout = render.layout_for(game.size, window)
        surface = pygame.Surface(layout.window_size)
        return render.draw_game(
            surface,
            game,
            layout,
            render.Fonts.default(),
            input_text="",
            controls=controls,
            explanation=EXPLANATION,
            explained=explained,
        )

    def panel(
        self,
        window: tuple[int, int] = render.DEFAULT_WINDOW,
        message: str | None = None,
        message_color: tuple[int, int, int] = render.TEXT,
        game: Game | None = None,
        **controls: object,
    ) -> tuple[pygame.Surface, dict[str, pygame.Rect]]:
        layout = render.layout_for(5, window)
        surface = pygame.Surface(layout.window_size)
        buttons = render.draw_panel(
            surface,
            game or Game(5),
            layout,
            render.Fonts.default(),
            message,
            message_color,
            input_text="",
            controls=render.Controls(**controls),  # type: ignore[arg-type]
        )
        return surface, buttons

    @pytest.mark.parametrize("window", [render.MIN_WINDOW, render.DEFAULT_WINDOW, (1920, 1080)])
    def test_why_sits_just_above_the_step_buttons(self, window: tuple[int, int]) -> None:
        _, buttons = self.panel(window, explains=True, explainable=True)
        why = buttons[render.EXPLAIN_BUTTON]
        assert render.layout_for(5, window).panel_rect.contains(why)
        for name, rect in buttons.items():
            if name != render.EXPLAIN_BUTTON:
                assert not why.colliderect(rect), name
        for name in render.NAV_BUTTONS:
            assert 0 < buttons[name].top - why.bottom <= 12  # right below it

    def test_why_stays_put_whatever_the_text_says(self) -> None:
        places = {
            self.panel(message=message, last_move=last, versus=versus, **state)[1][
                render.EXPLAIN_BUTTON
            ].topleft
            for message in (None, "Learner played 3c4>12. Tak!")
            for last in (None, "3c4>12")
            for versus in (None, "You play White against Learner")
            for state in (
                {"explains": True},
                {"explains": True, "explainable": True},
                {"explains": True, "explainable": True, "explaining": True},
            )
        }
        assert len(places) == 1

    def test_no_why_against_a_computer_that_cant_explain(self) -> None:
        _, buttons = self.panel(last_move="c3")
        assert render.EXPLAIN_BUTTON not in buttons

    def test_short_of_room_the_message_beats_who_plays_whom(
        self, started: Callable[..., Game]
    ) -> None:
        marker = (255, 0, 255)  # a color nothing else uses
        surface, _ = self.panel(
            render.MIN_WINDOW,
            game=started("a1 e5 b1 c3"),
            message="Learner played c3. Tak!",
            message_color=marker,
            last_move="c3",
            versus="You play White against Learner",
            explains=True,
        )
        panel = surface.subsurface(render.layout_for(5, render.MIN_WINDOW).panel_rect)
        drawn = pygame.mask.from_threshold(panel, marker, (1, 1, 1, 255)).count()
        assert drawn > 50  # the message is there

    @pytest.mark.parametrize("window", [render.MIN_WINDOW, render.DEFAULT_WINDOW, (1920, 1080)])
    def test_the_card_sits_over_the_board(self, window: tuple[int, int]) -> None:
        drawn = self.draw(window)
        card = drawn.buttons[render.EXPLANATION_CARD]
        assert drawn.viewport.contains(card)
        assert drawn.button_at(card.center) == render.EXPLANATION_CARD  # a click closes it

    def test_the_card_keeps_clear_of_the_move_it_explains(self) -> None:
        bottom = self.draw().buttons[render.EXPLANATION_CARD]
        # Squares near the front (bottom) of the board are under the card there...
        front = tuple(Square(file, 0) for file in range(3))
        moved = self.draw(explained=front).buttons[render.EXPLANATION_CARD]
        assert moved.top < bottom.top  # ...so it moves up out of the way
        # Squares at the back stay clear of it where it usually goes.
        back = tuple(Square(file, 4) for file in range(3))
        assert self.draw(explained=back).buttons[render.EXPLANATION_CARD] == bottom


class TestEvalBar:
    @pytest.mark.parametrize("window", [render.MIN_WINDOW, render.DEFAULT_WINDOW, (1920, 1080)])
    def test_the_bar_runs_down_the_left(self, window: tuple[int, int]) -> None:
        plain = render.layout_for(5, window)
        layout = render.layout_for(5, window, eval_bar=True)
        bar = layout.bar_rect
        assert plain.bar_rect is None and bar is not None
        assert pygame.Rect((0, 0), layout.window_size).contains(bar)
        assert bar.right < layout.viewport.left  # the board moves over to make room
        assert bar.height == layout.viewport.height
        assert layout.panel_rect == plain.panel_rect  # the panel stays where it was
        assert not layout.viewport.colliderect(layout.panel_rect)

    def draw(self, white: float, near: Color = Color.WHITE) -> tuple[pygame.Surface, pygame.Rect]:
        rect = pygame.Rect(10, 10, render.BAR_WIDTH, 400)
        surface = pygame.Surface((100, 420))
        label = f"{abs(white):.2f}"
        render.draw_eval_bar(
            surface, rect, render.Fonts.default(), Assessment(white, label, ""), near
        )
        return surface, rect

    def wood_at(self, surface: pygame.Surface, rect: pygame.Rect, fraction: float) -> Color:
        """Whose wood the bar shows at ``fraction`` of the way down it (off the label)."""
        pixel = tuple(surface.get_at((rect.left + 4, rect.top + round(fraction * rect.height))))
        light = sum(pixel[:3]) > sum(render.BAR_COLORS[Color.BLACK]) + 150
        return Color.WHITE if light else Color.BLACK

    def test_the_split_follows_the_score(self) -> None:
        even, rect = self.draw(0.0)
        assert self.wood_at(even, rect, 0.3) is Color.BLACK
        assert self.wood_at(even, rect, 0.7) is Color.WHITE
        white_ahead, rect = self.draw(0.6)  # White has 80% of the bar
        assert self.wood_at(white_ahead, rect, 0.3) is Color.WHITE
        black_ahead, rect = self.draw(-0.6)
        assert self.wood_at(black_ahead, rect, 0.7) is Color.BLACK

    def test_the_side_nearest_the_viewer_is_at_the_bottom(self) -> None:
        surface, rect = self.draw(0.6, near=Color.BLACK)  # seen from Black's side
        assert self.wood_at(surface, rect, 0.95) is Color.BLACK
        assert self.wood_at(surface, rect, 0.4) is Color.WHITE

    def test_a_win_fills_the_bar(self) -> None:
        surface, rect = self.draw(-1.0)
        assert all(self.wood_at(surface, rect, f) is Color.BLACK for f in (0.2, 0.5, 0.97))

    def test_drawn_with_the_game_when_there_is_room(self) -> None:
        layout = render.layout_for(5, eval_bar=True)
        surface = pygame.Surface(layout.window_size)
        fonts = render.Fonts.default()
        reading = Assessment(1.0, "Win", "", decided=True)
        render.draw_game(surface, Game(5), layout, fonts, assessment=reading)
        bar = layout.bar_rect
        assert bar is not None
        assert surface.get_at((bar.left + 4, bar.top + 40))[:3] == render.BAR_COLORS[Color.WHITE]
