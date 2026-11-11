"""Tests for the GUI's command line and window loop."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pygame
import pytest

from tak.engine import Game
from tak.gui import app


def scripted_events(monkeypatch: pytest.MonkeyPatch, events: list[pygame.event.Event]) -> None:
    """Make the window loop receive ``events``, in order, instead of real input."""
    queue: Iterator[pygame.event.Event] = iter(events)
    monkeypatch.setattr(pygame.event, "wait", lambda *args: next(queue))


@pytest.mark.parametrize(
    "closing",
    [
        pygame.event.Event(pygame.QUIT),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
    ],
)
def test_the_window_closes_on_quit_or_escape(
    monkeypatch: pytest.MonkeyPatch, closing: pygame.event.Event
) -> None:
    scripted_events(
        monkeypatch,
        [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a),  # ignored
            pygame.event.Event(pygame.VIDEORESIZE, w=600, h=400, size=(600, 400)),
            closing,
        ],
    )
    app.run(Game(4), window_size=(800, 600))
    assert not pygame.display.get_init()  # cleaned up afterwards


def test_main_opens_a_board_of_the_requested_size(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[Game] = []
    monkeypatch.setattr(app, "run", lambda game: shown.append(game))
    assert app.main(["--size", "7"]) == 0
    assert app.main(["-s", "3"]) == 0
    assert app.main([]) == 0
    assert [game.size for game in shown] == [7, 3, app.DEFAULT_SIZE]
    assert all(game.ply == 0 for game in shown)


@pytest.mark.parametrize("size", ["2", "9", "five"])
def test_main_rejects_unsupported_sizes(size: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        app.main(["--size", size])
    assert exit_info.value.code == 2
    assert "--size" in capsys.readouterr().err


def test_main_explains_how_to_install_pygame_if_it_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "pygame", None)  # makes `import pygame` fail
    assert app.main(["--size", "5"]) == 1
    assert 'pip install -e ".[gui]"' in capsys.readouterr().err
