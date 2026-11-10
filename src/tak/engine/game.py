"""Game state: whose turn it is, reserves, history, applying and undoing moves."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from tak.engine.board import BOARD_CONFIG, Board, Square
from tak.engine.moves import Move, PlaceMove, SlideMove, legal_moves, regular_moves
from tak.engine.pieces import Color, Piece, PieceType
from tak.engine.rules import (
    GameOverError,
    GameResult,
    IllegalMoveError,
    end_of_game,
    find_road,
    slide_problem,
)


@dataclass(frozen=True)
class Reserve:
    """Pieces a player has not yet placed."""

    stones: int
    caps: int

    @property
    def total(self) -> int:
        return self.stones + self.caps


@dataclass(frozen=True)
class _Undo:
    """What changed, so a move can be taken back."""

    changes: tuple[tuple[Square, tuple[Piece, ...]], ...]  # square -> stack before
    reserves: dict[Color, Reserve]
    result: GameResult | None


@dataclass
class PlayedMove:
    move: Move
    flattened: bool  # True if a capstone flattened a wall on this move
    undo: _Undo = field(repr=False)


class Game:
    """A game of Tak from the empty board to a result.

    White is Player 1. On the first turn each player places a flat stone of
    the *opponent's* color, then White begins regular play.
    """

    def __init__(self, size: int = 5) -> None:
        self.board = Board(size)
        stones, caps = BOARD_CONFIG[size]
        self.reserves: dict[Color, Reserve] = {color: Reserve(stones, caps) for color in Color}
        self.history: list[PlayedMove] = []
        self.result: GameResult | None = None

    # --- State ---------------------------------------------------------------

    @property
    def size(self) -> int:
        return self.board.size

    @property
    def ply(self) -> int:
        """How many moves have been played."""
        return len(self.history)

    @property
    def turn_number(self) -> int:
        """The 1-based turn number; each turn is one move per player."""
        return self.ply // 2 + 1

    @property
    def to_move(self) -> Color:
        return Color.WHITE if self.ply % 2 == 0 else Color.BLACK

    @property
    def in_opening(self) -> bool:
        """True during each player's first move, which places an opponent's flat."""
        return self.ply < 2

    @property
    def is_over(self) -> bool:
        return self.result is not None

    def reserve(self, color: Color) -> Reserve:
        return self.reserves[color]

    def legal_moves(self) -> list[Move]:
        return legal_moves(self)

    # --- Threats ("Tak" and "Tinue") --------------------------------------------

    def winning_move(self, color: Color) -> Move | None:
        """A move that would win the game for ``color`` if they could move right now.

        Whose turn it actually is doesn't matter: the question is "if ``color``
        took a regular turn in the current position, could they win?" Any win
        counts: a road (including when both roads are completed, since the
        mover wins those), or ending the game ahead on flats by placing their
        last piece or filling the board. Returns ``None`` if there is no such
        move or the game is already over. The game itself is left unchanged.
        """
        if self.is_over:
            return None
        return next(_winning_moves(self.board, self.reserves, color), None)

    def threatens_win(self, color: Color) -> bool:
        """True if ``color`` has a winning move available in the current position.

        Asked about the player who just moved, this is whether their move
        makes "Tak": a threat to win on their next turn.
        """
        return self.winning_move(color) is not None

    def is_tinue(self, color: Color) -> bool:
        """True if ``color`` threatens a win that their opponent can't stop.

        That is, ``color`` threatens to win (see :meth:`threatens_win`) and
        whatever regular move the opponent makes in reply, ``color`` still has a
        winning move afterwards, or the reply itself hands ``color`` the game
        (say, by completing only ``color``'s road). A reply that wins the game
        for the opponent or ends it in a draw does stop the threat. Asked
        about the player who just moved, this is whether their move makes
        "Tinue". The game itself is left unchanged.
        """
        if self.is_over:
            return False
        # Winning moves found so far. Most replies leave one of them intact, so
        # trying these first saves searching every move after every reply.
        known = list(_winning_moves(self.board, self.reserves, color))
        if not known:
            return False
        opponent = color.opponent
        for reply in regular_moves(self.board, opponent, self.reserves[opponent]):
            board, reserves = _after(self.board, self.reserves, opponent, reply)
            result = _result(board, opponent, reserves)
            if result is not None:
                if result.winner is color:
                    continue  # the reply loses on the spot, so it stops nothing
                return False  # the opponent wins or draws instead
            answer = next(_winning_moves(board, reserves, color, first=known), None)
            if answer is None:
                return False  # this reply stops every win
            if answer not in known:
                known.append(answer)
        return True

    # --- Checking and playing moves -------------------------------------------

    def check(self, move: Move) -> None:
        """Raise :class:`IllegalMoveError` (with the reason) if ``move`` isn't legal now."""
        if self.is_over:
            raise GameOverError("the game is over")
        problem = (
            self._place_problem(move) if isinstance(move, PlaceMove) else self._slide_problem(move)
        )
        if problem is not None:
            raise IllegalMoveError(problem)

    def play(self, move: Move) -> None:
        """Check ``move`` and, if legal, apply it and update the result."""
        self.check(move)
        mover = self.to_move
        undo_reserves = dict(self.reserves)
        if isinstance(move, PlaceMove):
            changes = self._apply_place(move, mover)
            flattened = False
        else:
            changes, flattened = self._apply_slide(move)
        self.history.append(
            PlayedMove(move, flattened, _Undo(tuple(changes.items()), undo_reserves, None))
        )
        out_of_pieces = any(reserve.total == 0 for reserve in self.reserves.values())
        self.result = end_of_game(self.board, mover, out_of_pieces)

    def undo(self) -> Move:
        """Take back the last move and return it."""
        if not self.history:
            raise ValueError("there are no moves to undo")
        played = self.history.pop()
        for square, stack in played.undo.changes:
            self.board.set_stack(square, stack)
        self.reserves = dict(played.undo.reserves)
        self.result = played.undo.result
        return played.move

    # --- Validation ------------------------------------------------------------

    def _place_problem(self, move: PlaceMove) -> str | None:
        if not self.board.in_bounds(move.square):
            return f"{move.square} is not on the board"
        if self.board.stack(move.square):
            return f"{move.square} is already occupied"
        if self.in_opening:
            if move.piece is not PieceType.FLAT:
                return "on the first turn each player places one flat stone (no walls or capstones)"
            return None
        reserve = self.reserve(self.to_move)
        if move.piece is PieceType.CAP:
            if reserve.caps < 1:
                if BOARD_CONFIG[self.size][1] == 0:
                    return f"a {self.size}x{self.size} game has no capstones"
                return "you have no capstones left in reserve"
        elif reserve.stones < 1:
            return "you have no stones left in reserve"
        return None

    def _slide_problem(self, move: SlideMove) -> str | None:
        if self.in_opening:
            return "on the first turn you must place a flat stone; stacks can't be moved yet"
        return slide_problem(self.board, self.to_move, move)

    # --- Applying (callers must have checked legality) -------------------------

    def _apply_place(self, move: PlaceMove, mover: Color) -> dict[Square, tuple[Piece, ...]]:
        # In the opening the piece belongs to the opponent and comes from their reserve.
        owner = mover.opponent if self.in_opening else mover
        self.reserves[owner] = _spend(self.reserves[owner], move.piece)
        previous = self.board.stack(move.square)
        self.board.set_stack(move.square, (Piece(owner, move.piece),))
        return {move.square: previous}

    def _apply_slide(self, move: SlideMove) -> tuple[dict[Square, tuple[Piece, ...]], bool]:
        return _slide_onto(self.board, move)


def _spend(reserve: Reserve, piece: PieceType) -> Reserve:
    """``reserve`` after one ``piece`` has been taken from it."""
    if piece is PieceType.CAP:
        return Reserve(reserve.stones, reserve.caps - 1)
    return Reserve(reserve.stones - 1, reserve.caps)


def _slide_onto(board: Board, move: SlideMove) -> tuple[dict[Square, tuple[Piece, ...]], bool]:
    """Carry out a (legal) slide on ``board``.

    Returns the stacks of every square touched, as they were before the move,
    and whether a capstone flattened a wall.
    """
    origin = board.stack(move.start)
    carried = origin[-move.count :]
    changes: dict[Square, tuple[Piece, ...]] = {move.start: origin}
    board.set_stack(move.start, origin[: -move.count])

    flattened = False
    used = 0
    square = move.start
    for take in move.drops:
        square = square.step(move.direction)
        existing = board.stack(square)
        changes[square] = existing
        if existing and existing[-1].type is PieceType.WALL:
            # A capstone landing alone flattens the wall beneath it.
            wall = existing[-1]
            existing = (*existing[:-1], Piece(wall.color, PieceType.FLAT))
            flattened = True
        board.set_stack(square, existing + carried[used : used + take])
        used += take
    return changes, flattened


# --- Looking ahead (for spotting threats) --------------------------------------
#
# These work on a board and reserves rather than a Game, so they can explore
# positions that never happen, like a player taking two turns in a row. They
# only cover regular turns (not the opening) and never change their inputs.

Reserves = dict[Color, Reserve]


def _after(board: Board, reserves: Reserves, color: Color, move: Move) -> tuple[Board, Reserves]:
    """The position after ``color`` makes ``move``, which must be legal."""
    board = board.copy()
    reserves = dict(reserves)
    if isinstance(move, PlaceMove):
        board.set_stack(move.square, (Piece(color, move.piece),))
        reserves[color] = _spend(reserves[color], move.piece)
    else:
        _slide_onto(board, move)
    return board, reserves


def _result(board: Board, mover: Color, reserves: Reserves) -> GameResult | None:
    """The result once ``mover`` has moved to reach this position, or ``None``."""
    out_of_pieces = any(reserve.total == 0 for reserve in reserves.values())
    return end_of_game(board, mover, out_of_pieces)


def _wins(board: Board, color: Color, reserves: Reserves) -> bool:
    """Whether ``color`` has won, having just moved to reach this position."""
    if find_road(board, color) is not None:
        return True  # the mover wins even if the opponent also has a road
    out_of_pieces = any(reserve.total == 0 for reserve in reserves.values())
    if not (out_of_pieces or board.is_full()):
        return False  # nothing ends the game, so there is no flat count
    result = end_of_game(board, color, out_of_pieces)
    return result is not None and result.winner is color


def _is_legal(board: Board, reserves: Reserves, color: Color, move: Move) -> bool:
    """Whether ``color`` could make ``move`` on a regular turn."""
    if isinstance(move, SlideMove):
        return slide_problem(board, color, move) is None
    if not board.in_bounds(move.square) or board.stack(move.square):
        return False
    reserve = reserves[color]
    return reserve.caps > 0 if move.piece is PieceType.CAP else reserve.stones > 0


def _winning_moves(
    board: Board, reserves: Reserves, color: Color, first: Iterable[Move] = ()
) -> Iterator[Move]:
    """Every move that would win for ``color`` right now.

    Moves in ``first`` are tried before the rest (they may be reported twice).
    Every move is only listed once the moves in ``first`` have been tried, so
    a caller that stops at the first win often never needs the full list.
    """
    for move in first:
        if _is_legal(board, reserves, color, move) and _wins_with(board, reserves, color, move):
            yield move
    for move in regular_moves(board, color, reserves[color]):
        if _wins_with(board, reserves, color, move):
            yield move


def _wins_with(board: Board, reserves: Reserves, color: Color, move: Move) -> bool:
    """Whether ``move`` (which must be legal) wins the game for ``color``."""
    after_board, after_reserves = _after(board, reserves, color, move)
    return _wins(after_board, color, after_reserves)
