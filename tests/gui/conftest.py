"""The GUI tests need pygame (an optional extra) and draw without opening a window."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
pytest.importorskip("pygame", reason="the GUI needs pygame: pip install -e '.[gui]'")
