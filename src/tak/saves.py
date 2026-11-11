"""Saving finished games as PTN files in the saves folder.

Both front ends (the terminal and the window) save a game here when it ends.
Each game gets its own new file, named after when it finished and how, for
example ``2026-11-09_14-03-22_5x5_white-road.ptn``. An existing file is never
overwritten: if the name is taken, ``-2``, ``-3``, ... is added.

The files are ordinary PTN, so :func:`tak.engine.game_from_ptn` (or any other
Tak tool) can load them again.

Where saves go, in order of preference:

1. a directory passed in explicitly (the ``--saves-dir`` option);
2. the ``TAK_SAVES_DIR`` environment variable;
3. the project's ``saves/`` folder, when running from a checkout of the project;
4. a ``saves/`` folder in the current directory, otherwise.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from tak.engine.game import Game
from tak.engine.notation import game_to_ptn
from tak.engine.rules import Reason

#: Environment variable that overrides the default saves folder.
SAVES_ENV = "TAK_SAVES_DIR"

SUFFIX = ".ptn"

# src/tak/saves.py -> the project root is two folders above the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_saves_dir() -> Path:
    """The folder games are saved to when none is given explicitly."""
    if override := os.environ.get(SAVES_ENV):
        return Path(override)
    if (_PROJECT_ROOT / "pyproject.toml").is_file():
        return _PROJECT_ROOT / "saves"
    return Path.cwd() / "saves"


def outcome_slug(game: Game) -> str:
    """A short filename-safe description of how the game ended, e.g. ``white-road``."""
    result = game.result
    if result is None:
        return "unfinished"
    if result.winner is None:
        return "draw"
    how = "road" if result.reason is Reason.ROAD else "flats"
    return f"{result.winner.value}-{how}"


def save_name(game: Game, when: datetime) -> str:
    """The file name (without any ``-2`` suffix) for ``game`` finished at ``when``."""
    stamp = when.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_{game.size}x{game.size}_{outcome_slug(game)}{SUFFIX}"


def game_text(game: Game, when: datetime) -> str:
    """The PTN written to a save file: the game plus when it was played."""
    return game_to_ptn(game, {"Date": when.strftime("%Y.%m.%d"), "Time": when.strftime("%H:%M:%S")})


def save_game(
    game: Game,
    directory: Path | None = None,
    *,
    when: datetime | None = None,
    replace: Path | None = None,
) -> Path:
    """Write ``game`` as PTN and return the path of the file written.

    A new file is created in ``directory`` (default: :func:`default_saves_dir`,
    created if missing), never overwriting an existing one. Pass ``replace``
    to rewrite a file saved earlier instead, for when the same game is
    finished again after an undo. Raises :class:`OSError` if the file can't be
    written.
    """
    when = when or datetime.now()
    text = game_text(game, when)
    if replace is not None:
        replace.write_text(text, encoding="utf-8")
        return replace

    folder = directory if directory is not None else default_saves_dir()
    folder.mkdir(parents=True, exist_ok=True)
    name = save_name(game, when)
    stem = name.removesuffix(SUFFIX)
    for attempt in range(1, 1000):
        path = folder / (name if attempt == 1 else f"{stem}-{attempt}{SUFFIX}")
        try:
            # "x" creates the file and fails if it already exists, so two
            # saves in the same second can never overwrite each other.
            with path.open("x", encoding="utf-8") as file:
                file.write(text)
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"too many saved games named {name} in {folder}")


def describe_path(path: Path) -> str:
    """``path`` relative to the current directory when it is inside it, for messages."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def save_finished_game(
    game: Game, directory: Path | None, replace: Path | None = None
) -> tuple[Path | None, str]:
    """Save ``game`` and return the path (``None`` on failure) and a line for the player.

    Unlike :func:`save_game` this never raises for a file problem: losing the
    save shouldn't crash the program at the end of a game, so the problem is
    reported in the message instead.
    """
    try:
        path = save_game(game, directory, replace=replace)
    except OSError as exc:
        return None, f"Could not save the game: {exc}"
    return path, f"Game saved to {describe_path(path)}"


def add_save_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``--saves-dir`` and ``--no-save`` options, shared by the terminal and the window."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--saves-dir",
        type=Path,
        metavar="DIR",
        help=f"folder to save finished games in (default: {describe_path(default_saves_dir())}, "
        f"or ${SAVES_ENV} if set)",
    )
    group.add_argument(
        "--no-save",
        action="store_true",
        help="don't save the game when it ends",
    )


def saves_dir_from(args: argparse.Namespace) -> Path | None:
    """Where to save, from parsed :func:`add_save_arguments` options; ``None`` to not save."""
    if args.no_save:
        return None
    directory: Path = args.saves_dir or default_saves_dir()
    return directory
