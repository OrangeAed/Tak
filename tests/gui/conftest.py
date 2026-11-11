"""Shared setup for the GUI tests: run pygame headless, skip if it's missing."""

from __future__ import annotations

import os

import pytest

pygame = pytest.importorskip("pygame")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="session", autouse=True)
def _pygame() -> object:
    pygame.init()
    yield pygame
    pygame.quit()
