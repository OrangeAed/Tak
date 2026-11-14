"""Looking one move ahead, quickly: immediate wins, safe moves, and Tinue.

The engine can already answer "could this player win right now?"
(:meth:`Game.winning_move`) and "is this Tinue?" (:meth:`Game.is_tinue`) by
trying every legal move. The computer players ask these questions after every
move they consider, hundreds of times a turn, so this module answers them
faster while giving the same answers:

- **Roads by placing a stone.** The player's road pieces are split into
  connected groups, each noting which board edges it touches. A stone placed
  on an empty square completes a road exactly when it joins groups that
  between them (with the square itself) touch two opposite edges. That takes
  a glance at each empty square, not a whole road search.
- **Roads by moving a stack.** A slide only changes the squares in one straight
  line from the stack. If the player could not make a road even by owning
  that whole line, no slide along it can win, so it is skipped. Only the few
  lines that pass this test have their slides tried one by one.
- **Flat wins.** These are only possible when a move could end the game
  without a road: when the board is nearly full or a player is down to their
  last piece. In those positions this module simply asks the engine.

Everything here takes a :class:`Game` and leaves it as it found it. Functions
that try moves play them on the game and take them back, so pass a copy
(:meth:`Game.copy`) if another thread could be looking at the game.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Iterator

from tak.engine.board import Board, Direction, Square
from tak.engine.game import Game
from tak.engine.moves import Move, PlaceMove, SlideMove
from tak.engine.pieces import Color, Piece, PieceType
from tak.engine.rules import drop_problem, slide_problem

# Which board edges a square (or a group of squares) touches, as bit flags.
LEFT, RIGHT, BOTTOM, TOP = 1, 2, 4, 8


def spans(flags: int) -> bool:
    """Whether edge ``flags`` include two opposite edges, i.e. make a road."""
    return (flags & (LEFT | RIGHT)) == (LEFT | RIGHT) or (flags & (BOTTOM | TOP)) == (BOTTOM | TOP)


# --- Board geometry, cached per size ------------------------------------------------
#
# Squares are numbered rank by rank from a1, like Board.squares():
# index = rank * size + file.


@functools.cache
def squares_of(size: int) -> tuple[Square, ...]:
    return tuple(Square(file, rank) for rank in range(size) for file in range(size))


@functools.cache
def neighbors_of(size: int) -> tuple[tuple[int, ...], ...]:
    """For each square index, the indices of its orthogonal neighbours."""
    result = []
    for square in squares_of(size):
        near = []
        for direction in Direction:
            other = square.step(direction)
            if 0 <= other.file < size and 0 <= other.rank < size:
                near.append(other.rank * size + other.file)
        result.append(tuple(near))
    return tuple(result)


@functools.cache
def edges_of(size: int) -> tuple[int, ...]:
    """For each square index, the edges it lies on."""
    last = size - 1
    result = []
    for square in squares_of(size):
        flags = 0
        if square.file == 0:
            flags |= LEFT
        if square.file == last:
            flags |= RIGHT
        if square.rank == 0:
            flags |= BOTTOM
        if square.rank == last:
            flags |= TOP
        result.append(flags)
    return tuple(result)


@functools.cache
def rays_of(size: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """For each square index and each :class:`Direction` (in order), the squares beyond it."""
    result = []
    for square in squares_of(size):
        per_direction = []
        for direction in Direction:
            line = []
            other = square.step(direction)
            while 0 <= other.file < size and 0 <= other.rank < size:
                line.append(other.rank * size + other.file)
                other = other.step(direction)
            per_direction.append(tuple(line))
        result.append(tuple(per_direction))
    return tuple(result)


def index(square: Square, size: int) -> int:
    return square.rank * size + square.file


Stacks = list[tuple[Piece, ...]]


def stacks_of(board: Board) -> Stacks:
    """Every stack on the board, indexed like :func:`squares_of`, for quick repeated lookups."""
    return [board.stack(square) for square in squares_of(board.size)]


# --- Road groups --------------------------------------------------------------------


def road_cells(stacks: Stacks, color: Color) -> list[bool]:
    """For each square, whether its top piece counts toward a road for ``color``."""
    return [
        bool(stack) and stack[-1].color is color and stack[-1].type is not PieceType.WALL
        for stack in stacks
    ]


def groups(cells: list[bool], size: int) -> tuple[list[int], list[int]]:
    """Connected groups of the ``True`` cells.

    Returns each square's group number (``-1`` where the cell is ``False``)
    and, for each group, the edges it touches.
    """
    near = neighbors_of(size)
    edges = edges_of(size)
    label = [-1] * len(cells)
    flags: list[int] = []
    for start, filled in enumerate(cells):
        if not filled or label[start] >= 0:
            continue
        group = len(flags)
        touched = 0
        label[start] = group
        todo = [start]
        while todo:
            cell = todo.pop()
            touched |= edges[cell]
            for other in near[cell]:
                if cells[other] and label[other] < 0:
                    label[other] = group
                    todo.append(other)
        flags.append(touched)
    return label, flags


def has_road(cells: list[bool], size: int) -> bool:
    """Whether the ``True`` cells contain a road (a group joining opposite edges)."""
    return any(spans(flags) for flags in groups(cells, size)[1])


# --- Trying moves without playing them ----------------------------------------------


def tops_after_slide(board: Board, move: SlideMove) -> dict[Square, Piece | None]:
    """The new top piece of every square a (legal) slide touches."""
    stack = board.stack(move.start)
    carried = stack[-move.count :]
    left_behind = stack[: -move.count]
    tops: dict[Square, Piece | None] = {move.start: left_behind[-1] if left_behind else None}
    used = 0
    square = move.start
    for take in move.drops:
        square = square.step(move.direction)
        used += take
        tops[square] = carried[used - 1]  # the last stone dropped here ends up on top
    return tops


def road_after(board: Board, color: Color, move: Move, cells: list[bool]) -> bool:
    """Whether ``color`` has a road after making ``move`` (which must be legal).

    ``cells`` is :func:`road_cells` for the current position; it isn't changed.
    """
    size = board.size
    after = list(cells)
    if isinstance(move, PlaceMove):
        if move.piece is PieceType.WALL:
            return has_road(after, size)
        after[index(move.square, size)] = True
    else:
        for square, top in tops_after_slide(board, move).items():
            after[index(square, size)] = (
                top is not None and top.color is color and top.counts_for_roads
            )
    return has_road(after, size)


def slides_along(board: Board, start: Square, direction: Direction) -> Iterator[SlideMove]:
    """Every legal slide of the stack on ``start`` in ``direction``.

    Whoever controls the stack is assumed to be the one moving it.
    """
    stack = board.stack(start)
    for count in range(1, min(len(stack), board.size) + 1):
        yield from _extend(board, start, direction, stack[-count:], 0, start, ())


def _extend(
    board: Board,
    origin: Square,
    direction: Direction,
    carried: tuple[Piece, ...],
    used: int,
    position: Square,
    drops: tuple[int, ...],
) -> Iterator[SlideMove]:
    square = position.step(direction)
    if not board.in_bounds(square):
        return
    destination_top = board.top(square)
    remaining = len(carried) - used
    for take in range(1, remaining + 1):
        is_last = take == remaining
        if drop_problem(square, destination_top, carried[used : used + take], is_last):
            continue
        if is_last:
            yield SlideMove(origin, direction, (*drops, take))
        else:
            yield from _extend(
                board, origin, direction, carried, used + take, square, (*drops, take)
            )


def _reach(stacks: Stacks, size: int, start: int, direction: int) -> list[int]:
    """Every square a slide from ``start`` in ``direction`` could possibly reach."""
    stack = stacks[start]
    cap_on_top = stack[-1].type is PieceType.CAP
    cells = []
    for cell in rays_of(size)[start][direction][: min(len(stack), size)]:
        target = stacks[cell]
        if target:
            kind = target[-1].type
            if kind is PieceType.CAP:
                break
            if kind is PieceType.WALL:
                if cap_on_top:
                    cells.append(cell)  # a capstone alone can flatten it, and stop there
                break
        cells.append(cell)
    return cells


def touched_squares(move: Move) -> list[Square]:
    """Every square a move changes: where a piece is placed, or a slide's whole path."""
    if isinstance(move, PlaceMove):
        return [move.square]
    return [move.start.step(move.direction, i) for i in range(len(move.drops) + 1)]


def is_legal(game: Game, color: Color, move: Move) -> bool:
    """Whether ``color`` could make ``move`` on a regular turn in this position."""
    board = game.board
    if isinstance(move, SlideMove):
        return slide_problem(board, color, move) is None
    if not board.in_bounds(move.square) or board.stack(move.square):
        return False
    reserve = game.reserve(color)
    return reserve.caps > 0 if move.piece is PieceType.CAP else reserve.stones > 0


# --- Immediate wins -----------------------------------------------------------------


def flat_win_possible(game: Game, color: Color, stacks: Stacks | None = None) -> bool:
    """Whether a move by ``color`` could end the game without a road.

    That needs the board to fill up or a player to run out of pieces. A
    placement fills one square and a slide at most ``size`` squares (the carry
    limit), and only a placement uses up a piece.
    """
    if stacks is None:
        stacks = stacks_of(game.board)
    empty = sum(1 for stack in stacks if not stack)
    if empty <= game.size:
        return True
    return game.reserve(color).total <= 1 or any(r.total == 0 for r in game.reserves.values())


def find_win(game: Game, color: Color, first: Iterable[Move] = ()) -> Move | None:
    """A move that would win the game for ``color`` right now, or ``None``.

    The same question as :meth:`Game.winning_move` (whose turn it is doesn't
    matter), answered faster. Moves in ``first`` are tried before anything
    else: pass wins found earlier, as they often still work.
    """
    if game.is_over:
        return None
    board = game.board
    stacks = stacks_of(board)
    if flat_win_possible(game, color, stacks):
        return game.winning_move(color)
    size = board.size
    cells = road_cells(stacks, color)
    label, flags = groups(cells, size)
    if any(spans(f) for f in flags):
        return game.winning_move(color)  # not a reachable position; let the engine decide

    for move in first:
        if is_legal(game, color, move) and road_after(board, color, move, cells):
            return move

    edges = edges_of(size)
    near = neighbors_of(size)
    squares = squares_of(size)

    def joined(cells_in: Iterable[int]) -> int:
        """Edges touched by ``cells_in`` together with every group next to them."""
        touched = 0
        for cell in cells_in:
            touched |= edges[cell]
            if label[cell] >= 0:
                touched |= flags[label[cell]]
            for other in near[cell]:
                if label[other] >= 0:
                    touched |= flags[label[other]]
        return touched

    reserve = game.reserve(color)
    piece = PieceType.FLAT if reserve.stones else PieceType.CAP if reserve.caps else None
    if piece is not None:
        for cell, stack in enumerate(stacks):
            if not stack and spans(joined((cell,))):
                return PlaceMove(squares[cell], piece)

    for cell, stack in enumerate(stacks):
        if not stack or stack[-1].color is not color:
            continue
        for number, direction in enumerate(Direction):
            line = _reach(stacks, size, cell, number)
            # If even owning the start and the whole line can't make a road, no slide can.
            if not line or not spans(joined([cell, *line])):
                continue
            for move in slides_along(board, squares[cell], direction):
                if road_after(board, color, move, cells):
                    return move
    return None


# --- Safety and Tinue ---------------------------------------------------------------


def loses_or_allows_win(game: Game, move: Move, threats: list[Move] | None = None) -> bool:
    """Whether ``move``, by the player to move, loses or lets the opponent win next move.

    A move "loses" if it ends the game with the opponent winning (say, a slide
    that completes only the opponent's road). A draw does not count as losing.

    ``threats`` is a list of opponent wins found so far. They are tried first,
    since most moves don't stop them, and any new one found is added, so pass
    the same list when checking many moves in a row.
    """
    if threats is None:
        threats = []
    mover = game.to_move
    game.play(move)
    try:
        if game.result is not None:
            return game.result.winner is mover.opponent
        reply = find_win(game, mover.opponent, first=threats)
        if reply is not None and reply not in threats:
            threats.append(reply)
        return reply is not None
    finally:
        game.undo()


def is_tinue(game: Game, color: Color, hints: Iterable[Move] = ()) -> bool:
    """Whether ``color`` has a win the opponent can't stop, with the opponent to move.

    The same question as :meth:`Game.is_tinue`, answered faster. ``hints`` are
    moves that may win for ``color``; they are checked, not trusted.
    """
    if game.is_over:
        return False
    if game.to_move is not color.opponent or game.in_opening:
        return game.is_tinue(color)
    win = find_win(game, color, first=hints)
    if win is None:
        return False
    known = [win]
    # Replies that touch the winning move's squares are the likeliest to stop
    # it, so try them first: when the threat can be stopped, that is usually
    # found after a few replies instead of most of them.
    needed = set(touched_squares(win))
    replies = sorted(
        game.legal_moves(), key=lambda reply: needed.isdisjoint(touched_squares(reply))
    )
    for reply in replies:
        game.play(reply)
        try:
            if game.result is not None:
                if game.result.winner is color:
                    continue  # the reply hands the game over, so it stops nothing
                return False  # the opponent wins or draws instead
            answer = find_win(game, color, first=known)
            if answer is None:
                return False  # this reply stops every win
            if answer not in known:
                known.append(answer)
        finally:
            game.undo()
    return True
