"""The learning opponent: plays by an evaluation it taught itself through self-play."""

from __future__ import annotations

import random
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from tak.ai import tactics
from tak.ai.base import Agent, opening_move, position_key
from tak.ai.learning import Weights, default_weights, features, value
from tak.ai.tactics import (
    Stacks,
    flat_win_possible,
    has_road,
    index,
    is_tinue,
    road_cells,
    stacks_of,
    tops_after_slide,
)
from tak.engine.game import Game
from tak.engine.moves import Move, PlaceMove, regular_moves
from tak.engine.pieces import Color, Piece, PieceType


@dataclass
class Option:
    """A move considered, and what playing it would lead to."""

    move: Move
    safe: bool  # the opponent can't win straight after it (and it doesn't lose)
    value: float  # how good the position after it is for us, -1 to +1 (learned)
    win_next: Move | None  # a win we would threaten after it (Tak), if any
    features: list[float] | None  # the position's features (None if the game ended)


#: Why the learner played a move: the step of its turn that decided it (see
#: :class:`LearnerAgent`). ``lost`` means every move let the opponent win.
Reason = Literal["opening", "win", "tinue", "best", "lost"]


@dataclass(frozen=True)
class Decision:
    """How the learner chose its move in one position, for explaining it afterwards."""

    move: Move
    reason: Reason
    #: The option played, with what it leads to (``None`` in the opening and for a win).
    chosen: Option | None = None
    #: How many moves it had that didn't let the opponent win straight away.
    safe: int = 0
    #: Moves it liked better but passed over, each with the opponent's reply
    #: that would have made Tinue, best-looking first.
    passed_over: tuple[tuple[Option, Move], ...] = ()


class LearnerAgent(Agent):
    """Wins, blocks, plays for Tinue, and otherwise trusts what it learned.

    Each turn, in order:

    1. If a move wins the game right now, play it.
    2. Set aside moves after which the opponent could win straight away, so
       it blocks threats and never hands over a win it could avoid.
    3. Among the safe moves, if one leaves a threat the opponent can't stop
       (Tinue), play it.
    4. Otherwise play the safe move whose position the learned evaluation
       (:mod:`tak.ai.learning`) likes best. Ties are broken at random. If
       the opponent could answer it with Tinue, try the next best instead
       (up to :attr:`tinue_checks` moves in all).

    Steps 1 to 3 are the same sure-fire tactics the strategist uses, and the
    Tinue check in step 4 is one more of the same kind: they look only a move
    or two ahead and are never wrong, but they have nothing to say about most
    moves. The learning comes in there: which of the many quiet moves is
    best is judged by weights the agent worked out by playing a thousand
    games against itself (see :mod:`tak.ai.training`), not by rules someone
    wrote. With untrained weights, the same tactics lose almost every game
    to the greedy player and the strategist.

    On the first turn it puts the opponent's stone in a corner.
    """

    key = "learner"
    name = "Learner"
    description = "taught itself by self-play; wins, blocks, and plays for Tinue"

    #: How many of the best-looking safe moves to check, in turn, for letting
    #: the opponent reply with Tinue, before settling for the best-looking one.
    tinue_checks = 8

    def __init__(
        self,
        seed: int | None = None,
        weights: Weights | None = None,
        tinue_checks: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.weights = weights if weights is not None else default_weights()
        if tinue_checks is not None:
            self.tinue_checks = tinue_checks
        # Recent decisions by position, so a move can be explained without
        # working it out again. The window thinks in another thread, hence the lock.
        self._decisions: OrderedDict[str, Decision] = OrderedDict()
        self._lock = threading.Lock()

    #: How many recent decisions to remember.
    remembered = 256

    def choose_move(self, game: Game) -> Move:
        return self.decide(game).move

    def decide(self, game: Game) -> Decision:
        """The move :meth:`choose_move` plays in ``game``, with how it was chosen.

        Always the same move as :meth:`choose_move` (it's the same work, and
        recent positions are remembered, so asking after the move was played
        is quick). The game is left unchanged.
        """
        if game.is_over:
            raise ValueError("the game is over")
        key = position_key(game)
        with self._lock:
            known = self._decisions.get(key)
            if known is not None:
                self._decisions.move_to_end(key)
                return known
        decision = self._decide(game.copy(), self.rng_for(game))
        with self._lock:
            self._decisions[key] = decision
            while len(self._decisions) > self.remembered:
                self._decisions.popitem(last=False)
        return decision

    def pick(self, game: Game, rng: random.Random) -> Move:
        return self._decide(game, rng).move

    def _decide(self, game: Game, rng: random.Random) -> Decision:
        if game.in_opening:
            return Decision(opening_move(game, rng), "opening")
        win = find_win(game, game.to_move)
        if win is not None:
            return Decision(win, "win")
        options = self.consider(game, rng)
        chosen, reason, passed_over = self._choose(game, options)
        safe = sum(1 for option in options if option.safe)
        return Decision(chosen.move, reason, chosen, safe, passed_over)

    def consider(self, game: Game, rng: random.Random) -> list[Option]:
        """Every legal move, in a random order, with what it leads to.

        Assumes the player to move has no winning move (check that first).
        """
        moves = game.legal_moves()
        rng.shuffle(moves)  # so ties go to a random one of the best
        threats: list[Move] = []  # opponent wins seen so far, tried first next time
        return [self._consider(game, move, threats) for move in moves]

    def best(self, game: Game, options: list[Option]) -> Option:
        """The option to play.

        A safe move that makes Tinue if there is one; otherwise the safe move
        the learned evaluation likes best, passing over any after which the
        opponent could make Tinue themselves.
        """
        return self._choose(game, options)[0]

    def _choose(
        self, game: Game, options: list[Option]
    ) -> tuple[Option, Reason, tuple[tuple[Option, Move], ...]]:
        """:meth:`best`, also saying why, and which better-looking moves it passed over."""
        safe = [option for option in options if option.safe]
        if not safe:
            return max(options, key=lambda option: option.value), "lost", ()  # lost anyway
        me = game.to_move
        taks = sorted(
            (option for option in safe if option.win_next is not None),
            key=lambda option: option.value,
            reverse=True,
        )
        for option in taks:
            assert option.win_next is not None
            game.play(option.move)
            try:
                if is_tinue(game, me, hints=[option.win_next]):
                    return option, "tinue", ()
            finally:
                game.undo()
        # Best first; the sort keeps the random order among equals, so ties stay random.
        ranked = sorted(safe, key=lambda option: option.value, reverse=True)
        passed_over: list[tuple[Option, Move]] = []
        for option in ranked[: self.tinue_checks]:
            reply = tinue_reply(game, option.move)
            if reply is None:
                return option, "best", tuple(passed_over)
            passed_over.append((option, reply))
        return ranked[0], "best", ()  # every move checked allows Tinue: play the best-looking

    def _consider(self, game: Game, move: Move, threats: list[Move]) -> Option:
        """Play ``move`` on ``game``, look at the result, and take it back."""
        me = game.to_move
        game.play(move)
        try:
            result = game.result
            if result is not None:
                # Not a win (that was checked first): a draw is safe, a loss isn't.
                outcome = 0.0 if result.winner is None else 1.0 if result.winner is me else -1.0
                return Option(move, outcome >= 0, outcome, None, None)
            stacks = stacks_of(game.board)
            reply = find_win(game, me.opponent, first=threats)
            if reply is not None:
                if reply not in threats:
                    threats.append(reply)
                x = features(game, me, threat=False, stacks=stacks)
                # Losing next move; the value only matters if every move loses.
                return Option(move, False, value(self.weights, x) - 2.0, None, x)
            win_next = find_win(game, me)
            x = features(game, me, threat=win_next is not None, stacks=stacks)
            return Option(move, True, value(self.weights, x), win_next, x)
        finally:
            game.undo()


def allows_tinue(game: Game, move: Move) -> bool:
    """Whether, after ``move`` by the player to move, the opponent could reply with Tinue.

    That is, some reply leaves the opponent threatening a win that can't be
    stopped. ``move`` should be safe (it doesn't let the opponent win at once).
    """
    return tinue_reply(game, move) is not None


def tinue_reply(game: Game, move: Move) -> Move | None:
    """The opponent's reply to ``move`` that would make Tinue, or ``None`` if there is none.

    See :func:`allows_tinue`. The game is left unchanged.
    """
    me = game.to_move
    them = me.opponent
    game.play(move)
    try:
        if game.is_over:
            return None
        threats: list[Move] = []  # their wins seen so far, tried first next time
        for reply in game.legal_moves():
            game.play(reply)
            try:
                if game.result is not None:
                    continue  # a reply that ends the game makes no threat
                threat = find_win(game, them, first=threats)
                if threat is None:
                    continue
                if threat not in threats:
                    threats.append(threat)
                if is_tinue(game, them, hints=[threat]):
                    return reply
            finally:
                game.undo()
        return None
    finally:
        game.undo()


# --- Finding wins near the end of the game ---------------------------------------------


def find_win(game: Game, color: Color, first: Iterable[Move] = ()) -> Move | None:
    """A move that would win the game for ``color`` right now, or ``None``.

    The same answer as :func:`tak.ai.tactics.find_win` (a winning move
    exactly when there is one), but quicker near the end of the game. There,
    any move that fills the board or uses up a player's last piece ends the
    game on flat count, so the quick road-only search can't be used and
    :func:`tactics.find_win` asks the engine to play out every move. This
    tries every move too, but on plain lists instead of a board, which is
    several times faster. The learner asks this twice for every move it
    considers, so it makes a big difference to how long training takes.
    """
    if game.is_over:
        return None
    stacks = stacks_of(game.board)
    if not flat_win_possible(game, color, stacks):
        return tactics.find_win(game, color, first)
    return _endgame_win(game, color, stacks)


def _endgame_win(game: Game, color: Color, stacks: Stacks) -> Move | None:
    """Try every move for ``color``, judging each the way the engine ends a game."""
    board = game.board
    size = board.size
    them = color.opponent
    mine = road_cells(stacks, color)
    theirs = road_cells(stacks, them)
    tops: list[Piece | None] = [stack[-1] if stack else None for stack in stacks]
    empty = tops.count(None)

    def flat_lead(top: Piece | None) -> int:
        """What a top piece adds to ``color``'s lead on flats."""
        if top is None or top.type is not PieceType.FLAT:
            return 0
        return 1 if top.color is color else -1

    lead = sum(flat_lead(top) for top in tops)
    reserve = game.reserve(color)
    they_are_out = game.reserve(them).total == 0

    for move in regular_moves(board, color, reserve):
        changes: dict[int, Piece | None]
        if isinstance(move, PlaceMove):
            changes = {index(move.square, size): Piece(color, move.piece)}
            left = reserve.total - 1
        else:
            changes = {
                index(square, size): top for square, top in tops_after_slide(board, move).items()
            }
            left = reserve.total
        ours, others = list(mine), list(theirs)
        empty_after, lead_after = empty, lead
        # Neither player has a road yet, so one can only appear where a square turns theirs.
        ours_grew = others_grew = False
        for cell, top in changes.items():
            road = top is not None and top.type is not PieceType.WALL
            ours[cell] = road and top is not None and top.color is color
            others[cell] = road and top is not None and top.color is them
            ours_grew = ours_grew or (ours[cell] and not mine[cell])
            others_grew = others_grew or (others[cell] and not theirs[cell])
            empty_after += (top is None) - (tops[cell] is None)
            lead_after += flat_lead(top) - flat_lead(tops[cell])
        if ours_grew and has_road(ours, size):
            return move  # the mover wins even if the opponent also has a road
        if others_grew and has_road(others, size):
            continue  # completes only the opponent's road
        ends = left == 0 or they_are_out or empty_after == 0
        if ends and lead_after > 0:
            return move
    return None
