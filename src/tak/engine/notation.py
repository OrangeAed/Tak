"""Portable Tak Notation (PTN): parsing and writing moves and games.

See ``docs/notation.md``. This module only turns text into moves and back; it
knows nothing about whether a move is legal (that is the engine's job).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from tak.engine.board import Direction, Square
from tak.engine.game import Game
from tak.engine.moves import Move, PlaceMove, SlideMove
from tak.engine.pieces import PieceType
from tak.engine.rules import IllegalMoveError


class NotationError(ValueError):
    """Raised when text is not valid PTN."""


_ARROWS = str.maketrans({"←": "<", "→": ">", "↑": "+", "↓": "-"})
_ANNOTATIONS = re.compile(r"""[*'"!?]+$""")
_PLACE = re.compile(r"([FSC])?([a-h][1-8])")
_SLIDE = re.compile(r"([1-8])?([a-h][1-8])([<>+\-])([1-8]+)?([FSC])?")
_HEADER = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
_COMMENT = re.compile(r"\{[^}]*\}")
_MOVE_NUMBER = re.compile(r"^\d+\.")
_RESULTS = frozenset({"R-0", "0-R", "F-0", "0-F", "1-0", "0-1", "1/2-1/2"})

_PIECE_BY_LETTER = {piece.value: piece for piece in PieceType}
_DIRECTION_BY_SYMBOL = {direction.value: direction for direction in Direction}


def parse_move(text: str) -> Move:
    """Parse one PTN move such as ``a1``, ``Sb2``, ``Cc3`` or ``3c2>12``.

    Annotation marks (``*``, ``'``, ``!``, ``?``) are accepted and ignored, as
    are the arrow forms of the directions.
    """
    token = _ANNOTATIONS.sub("", text.strip().translate(_ARROWS))
    if not token:
        raise NotationError("empty move")

    if match := _PLACE.fullmatch(token):
        letter, square = match.groups()
        piece = _PIECE_BY_LETTER[letter] if letter else PieceType.FLAT
        return PlaceMove(Square.from_name(square), piece)

    if match := _SLIDE.fullmatch(token):
        count_text, square, symbol, drops_text, _top = match.groups()
        count = int(count_text) if count_text else 1
        drops = tuple(int(digit) for digit in drops_text) if drops_text else (count,)
        if sum(drops) != count:
            raise NotationError(
                f"in {token!r} the drop counts add up to {sum(drops)}, "
                f"but {count} stone(s) are picked up"
            )
        return SlideMove(Square.from_name(square), _DIRECTION_BY_SYMBOL[symbol], drops)

    raise NotationError(
        f"can't read {text.strip()!r} as a move. "
        "Examples: a1 (flat), Sb2 (wall), Cc3 (capstone), 3c2>12 (move a stack)"
    )


def format_move(move: Move, *, flattened: bool = False) -> str:
    """Write ``move`` in its shortest standard PTN form.

    Pass ``flattened=True`` to add the ``*`` mark for a capstone flattening a wall.
    """
    if isinstance(move, PlaceMove):
        letter = "" if move.piece is PieceType.FLAT else move.piece.value
        text = f"{letter}{move.square}"
    else:
        count = str(move.count) if move.count > 1 else ""
        drops = "".join(str(drop) for drop in move.drops) if len(move.drops) > 1 else ""
        text = f"{count}{move.start}{move.direction.value}{drops}"
    return text + "*" if flattened else text


@dataclass
class ParsedPtn:
    headers: dict[str, str] = field(default_factory=dict)
    moves: list[Move] = field(default_factory=list)
    result: str | None = None


def parse_ptn(text: str) -> ParsedPtn:
    """Split PTN text into headers, moves and the result token."""
    text = _COMMENT.sub(" ", text)
    parsed = ParsedPtn(headers=dict(_HEADER.findall(text)))
    body = _HEADER.sub(" ", text)
    for token in body.split():
        token = _MOVE_NUMBER.sub("", token)
        if not token or token == "...":
            continue
        if token in _RESULTS:
            parsed.result = token
        else:
            parsed.moves.append(parse_move(token))
    return parsed


def game_to_ptn(game: Game, headers: Mapping[str, str] | None = None) -> str:
    """Write ``game`` as PTN: headers, numbered moves, and the result if finished."""
    lines = {"Size": str(game.size)}
    if headers:
        lines.update(headers)
    if game.result is not None:
        lines["Result"] = game.result.ptn
    out = [f'[{key} "{value}"]' for key, value in lines.items()]

    tokens = [format_move(played.move, flattened=played.flattened) for played in game.history]
    rows = [f"{i // 2 + 1}. {' '.join(tokens[i : i + 2])}" for i in range(0, len(tokens), 2)]
    if game.result is not None and rows:
        rows[-1] += f" {game.result.ptn}"
    return "\n".join([*out, "", *rows]) + "\n"


def game_from_ptn(text: str) -> Game:
    """Replay PTN text into a :class:`Game`, checking every move for legality."""
    parsed = parse_ptn(text)
    if "TPS" in parsed.headers:
        raise NotationError("games that start from a TPS position are not supported yet")
    try:
        size = int(parsed.headers["Size"])
    except KeyError:
        raise NotationError('the PTN has no [Size "N"] header') from None
    except ValueError:
        raise NotationError(f"bad board size: {parsed.headers['Size']!r}") from None

    try:
        game = Game(size)
    except ValueError as exc:
        raise NotationError(str(exc)) from exc
    for number, move in enumerate(parsed.moves, start=1):
        try:
            game.play(move)
        except IllegalMoveError as exc:
            raise IllegalMoveError(f"move {number} ({format_move(move)}): {exc}") from exc
    return game
