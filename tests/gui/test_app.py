"""The window's controller (driven by synthetic pygame events) and the command line."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pygame
import pytest

from tak.engine import Color, Game, Piece, PieceType, Square, game_to_ptn, parse_move
from tak.gui import render
from tak.gui.app import DRAG_THRESHOLD, Controller, main
from tak.gui.session import BAD, GameSession

ROAD_WIN = ["a1", "c3", "b1", "b3", "b2", "a3", "c2"]


class Window:
    """A controller plus an off-screen surface to draw on, with input helpers."""

    def __init__(self, session: GameSession | None = None, size: tuple[int, int] | None = None):
        self.controller = Controller(session or GameSession(Game(5)), size)
        self.surface = pygame.Surface(self.controller.layout.window_size)
        self.draw()

    @property
    def game(self) -> Game:
        return self.controller.game

    def draw(self) -> None:
        self.controller.draw(self.surface)

    def send(self, kind: int, **attributes: object) -> None:
        self.controller.handle(pygame.event.Event(kind, **attributes))
        if self.controller.dirty:
            self.draw()

    def where(self, name: str) -> tuple[int, int]:
        square = Square.from_name(name)
        return self.controller.frame.to_window((square.file + 0.5, square.rank + 0.5, 0.0))

    def click(self, pos: tuple[int, int], button: int = 1) -> None:
        self.send(pygame.MOUSEBUTTONDOWN, button=button, pos=pos)
        self.send(pygame.MOUSEBUTTONUP, button=button, pos=pos)

    def click_square(self, *names: str) -> None:
        for name in names:
            self.click(self.where(name))

    def click_button(self, name: str) -> None:
        self.click(self.controller.frame.buttons[name].center)

    def key(self, key: int, mod: int = 0, unicode: str = "") -> None:
        self.send(pygame.KEYDOWN, key=key, mod=mod, unicode=unicode)

    def release(self, key: int) -> None:
        self.send(pygame.KEYUP, key=key, mod=0, unicode="")

    def type(self, text: str) -> None:
        for char in text:
            self.key(ord(char), unicode=char)
            self.release(ord(char))
        self.key(pygame.K_RETURN)
        self.release(pygame.K_RETURN)


def sq(name: str) -> Square:
    return Square.from_name(name)


class TestClickToMove:
    def test_click_an_empty_square_to_place_a_flat(self) -> None:
        window = Window()
        window.click_square("c3")
        assert window.game.history[-1].move == parse_move("c3")

    def test_move_a_stack_by_clicking(self) -> None:
        window = Window()
        for move in ["a1", "e5", "c3", "e4"]:
            window.type(move)
        window.click_square("c3", "c4")
        assert window.game.history[-1].move == parse_move("c3+")

    def test_right_click_cancels_a_lift(self) -> None:
        window = Window()
        for move in ["a1", "e5", "c3", "e4"]:
            window.type(move)
        window.click_square("c3")
        assert window.controller.builder.lift is not None
        window.click(window.where("c3"), button=3)
        assert window.controller.builder.lift is None

    def test_escape_cancels_a_lift(self) -> None:
        window = Window()
        for move in ["a1", "e5", "c3", "e4"]:
            window.type(move)
        window.click_square("c3")
        window.key(pygame.K_ESCAPE)
        assert window.controller.builder.lift is None

    def test_choose_a_wall_in_the_panel(self) -> None:
        window = Window()
        window.type("a1")
        window.type("e5")
        window.click_button("place:S")
        assert window.controller.place is PieceType.WALL
        window.click_square("c3")
        assert window.game.history[-1].move == parse_move("Sc3")
        assert window.controller.place is PieceType.FLAT  # back to flats afterwards

    @pytest.mark.parametrize("mods,piece", [(pygame.KMOD_SHIFT, "Sc3"), (pygame.KMOD_CTRL, "Cc3")])
    def test_shift_and_ctrl_click(self, mods: int, piece: str) -> None:
        window = Window()
        window.type("a1")
        window.type("e5")
        window.controller.click(window.where("c3"), mods)
        assert window.game.history[-1].move == parse_move(piece)

    def test_an_illegal_click_explains_itself(self) -> None:
        window = Window()
        window.type("a1")
        window.click_square("a1")  # Black's turn, in the opening: can't move stacks
        kind, _ = window.controller.message or ("", "")
        assert kind == BAD and window.game.ply == 1

    def test_clicking_a_tall_stack_picks_it_not_the_square_behind(self) -> None:
        window = Window()
        for move in ["a1", "e5"]:
            window.type(move)
        white = Piece(Color.WHITE, PieceType.FLAT)
        window.game.board.set_stack(sq("c2"), (white,) * 12)
        window.draw()
        # A point well up the stack's side, which is drawn over c3 and beyond.
        pos = window.controller.frame.to_window((2.5, 1.5, 1.2))
        assert window.controller.frame.square_at(pos) == sq("c2")


class TestMouse:
    def test_a_drag_turns_the_board_instead_of_clicking(self) -> None:
        window = Window()
        start = window.where("c3")
        camera = window.controller.camera
        window.send(pygame.MOUSEBUTTONDOWN, button=1, pos=start)
        end = (start[0] + 10 * DRAG_THRESHOLD, start[1])
        window.send(pygame.MOUSEMOTION, pos=end, rel=(10 * DRAG_THRESHOLD, 0), buttons=(1, 0, 0))
        window.send(pygame.MOUSEBUTTONUP, button=1, pos=end)
        assert window.controller.camera != camera
        assert window.game.ply == 0  # no piece placed

    def test_a_tiny_wobble_is_still_a_click(self) -> None:
        window = Window()
        start = window.where("c3")
        window.send(pygame.MOUSEBUTTONDOWN, button=1, pos=start)
        window.send(pygame.MOUSEMOTION, pos=(start[0] + 2, start[1]), rel=(2, 0), buttons=(1, 0, 0))
        window.send(pygame.MOUSEBUTTONUP, button=1, pos=(start[0] + 2, start[1]))
        assert window.game.ply == 1

    def test_hovering_tracks_the_square(self) -> None:
        window = Window()
        window.send(pygame.MOUSEMOTION, pos=window.where("b2"), rel=(0, 0), buttons=(0, 0, 0))
        assert window.controller.hover == sq("b2")

    def test_wheel_zooms(self) -> None:
        window = Window()
        window.send(pygame.MOUSEWHEEL, x=0, y=2)
        assert window.controller.camera.zoom > 1


class TestKeys:
    def test_holding_backspace_repeats(self) -> None:
        window = Window()
        for char in "3c3>12":
            window.key(ord(char), unicode=char)
        for _ in range(4):  # one press, then repeats (no key-up in between)
            window.key(pygame.K_BACKSPACE)
        assert window.controller.typed == "3c"

    def test_holding_tab_flips_only_once(self) -> None:
        window = Window()
        yaw = window.controller.camera.yaw
        for _ in range(5):
            window.key(pygame.K_TAB)
        assert window.controller.camera.yaw == pytest.approx((yaw + math.pi) % math.tau)
        window.release(pygame.K_TAB)
        window.key(pygame.K_TAB)
        assert window.controller.camera.yaw == pytest.approx(yaw)

    def test_holding_enter_does_not_confirm_by_accident(self, tmp_path: Path) -> None:
        path = tmp_path / "game.ptn"
        game = Game(3)
        for move in ROAD_WIN:
            game.play(parse_move(move))
        path.write_text(game_to_ptn(game), encoding="utf-8")
        window = Window(GameSession.load(path))
        for char in "b2":
            window.key(ord(char), unicode=char)
        window.key(pygame.K_RETURN)  # asks for confirmation
        window.key(pygame.K_RETURN)  # a repeat: ignored
        assert window.game.ply == 0 and window.controller.session.pending is not None
        window.release(pygame.K_RETURN)
        window.key(pygame.K_RETURN)  # a real second press confirms
        assert window.game.ply == 1

    def test_typed_moves(self) -> None:
        window = Window()
        window.type("a1")
        assert window.game.ply == 1 and window.controller.typed == ""

    def test_ctrl_q_quits(self) -> None:
        window = Window()
        window.key(pygame.K_q, mod=pygame.KMOD_CTRL)
        assert not window.controller.running


class TestStepping:
    @pytest.fixture
    def window(self) -> Window:
        window = Window(GameSession(Game(3)))
        for move in ROAD_WIN:
            window.type(move)
        return window

    def test_buttons(self, window: Window) -> None:
        window.click_button("nav:back")
        assert window.game.ply == 6
        window.click_button("nav:start")
        assert window.game.ply == 0
        window.click_button("nav:forward")
        assert window.game.ply == 1
        window.click_button("nav:end")
        assert window.game.ply == 7 and window.game.is_over

    @pytest.mark.parametrize(
        "back,forward", [(pygame.K_LEFT, pygame.K_RIGHT), (pygame.K_z, pygame.K_y)]
    )
    def test_ctrl_keys_step_and_repeat(self, window: Window, back: int, forward: int) -> None:
        for _ in range(3):  # held down: repeats keep stepping
            window.key(back, mod=pygame.KMOD_CTRL)
        assert window.game.ply == 4
        window.key(forward, mod=pygame.KMOD_CTRL)
        assert window.game.ply == 5

    def test_ctrl_home_and_end(self, window: Window) -> None:
        window.key(pygame.K_HOME, mod=pygame.KMOD_CTRL)
        assert window.game.ply == 0
        window.key(pygame.K_END, mod=pygame.KMOD_CTRL)
        assert window.game.ply == 7

    def test_plain_home_resets_the_view_not_the_game(self, window: Window) -> None:
        window.send(pygame.MOUSEWHEEL, x=0, y=3)
        window.key(pygame.K_HOME)
        assert window.game.ply == 7
        assert window.controller.camera == window.controller.default_camera

    def test_stepping_drops_a_half_made_move(self) -> None:
        window = Window()
        for move in ["a1", "e5", "c3", "e4"]:
            window.type(move)
        window.click_square("c3")
        window.click_button("nav:back")
        assert window.controller.builder.lift is None


class TestResizing:
    def test_the_layout_follows_the_window(self) -> None:
        window = Window()
        small = window.controller.layout.viewport.copy()
        window.controller.resize((1600, 1000))
        assert window.controller.layout.viewport.width > small.width
        assert window.controller.layout.viewport.height > small.height

    def test_never_smaller_than_the_minimum(self) -> None:
        window = Window()
        assert window.controller.resize((300, 200)) == render.MIN_WINDOW

    def test_drawing_on_a_resized_surface_adapts(self) -> None:
        window = Window()
        surface = pygame.Surface((1400, 900))
        window.controller.draw(surface)
        assert window.controller.layout.window_size == (1400, 900)
        assert window.controller.frame.viewport == window.controller.layout.viewport

    def test_clicks_still_land_after_resizing(self) -> None:
        window = Window()
        window.surface = pygame.Surface((1500, 950))
        window.draw()
        window.click_square("d2")
        assert window.game.history[-1].move == parse_move("d2")


class TestMain:
    @pytest.fixture
    def runs(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[GameSession, object]]:
        calls: list[tuple[GameSession, object]] = []
        monkeypatch.setattr(
            "tak.gui.app.run", lambda session, message: calls.append((session, message))
        )
        return calls

    def test_default_is_a_new_five_by_five_game(self, runs: list, saves_dir: Path) -> None:
        assert main([]) == 0
        [(session, _)] = runs
        assert session.game.size == 5 and session.game.ply == 0
        assert session.recorder.save_dir == saves_dir

    def test_size(self, runs: list) -> None:
        main(["--size", "6"])
        assert runs[0][0].game.size == 6

    def test_rejects_bad_size(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--size", "9"])
        assert "--size" in capsys.readouterr().err

    def test_load_opens_at_the_start(self, runs: list, tmp_path: Path) -> None:
        path = tmp_path / "game.ptn"
        game = Game(3)
        for move in ROAD_WIN:
            game.play(parse_move(move))
        path.write_text(game_to_ptn(game), encoding="utf-8")
        assert main(["--load", str(path)]) == 0
        [(session, message)] = runs
        assert session.game.ply == 0 and session.total == 7 and session.source == path
        assert "7 moves" in str(message)

    def test_load_errors_are_reported(
        self, runs: list, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--load", str(tmp_path / "missing.ptn")]) == 1
        assert "Could not open" in capsys.readouterr().err
        assert runs == []

    def test_load_and_size_exclude_each_other(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["--load", str(tmp_path / "x.ptn"), "--size", "5"])

    def test_save_options(self, runs: list, tmp_path: Path) -> None:
        main(["--saves-dir", str(tmp_path)])
        main(["--no-save"])
        assert runs[0][0].recorder.save_dir == tmp_path
        assert runs[1][0].recorder.save_dir is None


def test_run_as_a_module() -> None:
    """The window is started with ``python -m tak.gui``; check that entry point answers."""
    completed = subprocess.run(
        [sys.executable, "-m", "tak.gui", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "python -m tak.gui" in completed.stdout
    assert "--load" in completed.stdout and "--size" in completed.stdout
