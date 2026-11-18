"""Explaining the learner's moves in plain words.

The learner (:class:`tak.ai.learner_agent.LearnerAgent`) remembers how it
chose each move: which step of its turn decided it (a win, a block, Tinue, or
its learned evaluation) and which better-looking moves it passed over. An
explanation is built from that record, so it tells the truth about why the
move was played rather than guessing afterwards.

When the learned evaluation decided, the explanation says what the move
gained, in the evaluation's own terms. Each feature's change from before the
move to after it, times its learned weight, is how much that change counted
for the move. Related features are added up into a few things a player would
recognise (the flat count, road building, blocking, stacks, ...), and the
biggest gains are described using the board itself: counts of flats, how many
squares each player is from a road, and so on.

Nothing here draws anything; the window shows the :class:`Explanation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard

from tak.ai.base import Agent
from tak.ai.evaluation import endgame, longest_road, road_gap
from tak.ai.learner_agent import Decision, LearnerAgent, find_win
from tak.ai.learning import FEATURE_NAMES, features
from tak.ai.tactics import stacks_of
from tak.engine.game import Game
from tak.engine.notation import format_move
from tak.engine.pieces import Color, PieceType
from tak.engine.rules import Reason, count_flats

#: Most gains mentioned in one explanation.
MAX_POINTS = 3
#: Gains smaller than this (in the evaluation's units, before squashing to
#: -1..+1) aren't worth mentioning. A flat more on 5x5 is worth about 0.1.
MIN_GAIN = 0.01

#: Features that together make up each thing a gain can be about.
GROUPS: dict[str, tuple[str, ...]] = {
    "flats": ("flats", "flats_late"),
    "roads": (
        "road_gap",
        "longest_road",
        "lines",
        "road_squares",
        "our_gap_2",
        "our_gap_3",
        "their_gap_2",
        "their_gap_3",
    ),
    "stacks": ("captives", "prisoners"),
    "center": ("center",),
    "walls": ("walls",),
    "capstone": ("caps",),
    "reserves": ("reserves",),
}


@dataclass(frozen=True)
class Explanation:
    """Why a move was played, in a few sentences."""

    #: For example ``Why Learner played c3``.
    title: str
    #: The main reason, in a sentence or two.
    summary: str
    #: What the move gained, most important first (may be empty).
    points: tuple[str, ...] = ()
    #: How the player rates its position after the move, if that's worth saying.
    outlook: str | None = None


@dataclass(frozen=True)
class Opponent:
    """How to refer to the player on the other side: "you", or by color."""

    subject: str  # "you" / "White"
    possessive: str  # "your" / "White's"
    person: bool  # "you" takes "are", "have", "need"; a color takes "is", "has", "needs"

    @classmethod
    def of(cls, color: Color, person: bool) -> Opponent:
        if person:
            return cls("you", "your", True)
        return cls(str(color), f"{color}'s", False)

    def verb(self, plural: str, singular: str) -> str:
        """``subject`` with the right form of a verb, e.g. "you are" or "White is"."""
        return f"{self.subject} {plural if self.person else singular}"


def can_explain(agent: Agent | None) -> TypeGuard[LearnerAgent]:
    """Whether :func:`explain_move` can explain moves by ``agent``."""
    return isinstance(agent, LearnerAgent)


def explain_move(agent: LearnerAgent, game: Game, person: bool = True) -> Explanation:
    """Why ``agent`` plays the move it does in ``game``, the position before the move.

    ``person`` says whether its opponent is a person, who is then called
    "you"; otherwise the opponent is named by color. The game is left
    unchanged. This is quick for a move the agent has just chosen, since it
    remembers how it chose it.
    """
    before = game.copy()
    decision = agent.decide(before)
    me = before.to_move
    them = Opponent.of(me.opponent, person)
    move_text = format_move(decision.move)
    after = before.copy()
    after.play(decision.move)
    played = after.history[-1]
    title = f"Why {agent.name} played {format_move(played.move, flattened=played.flattened)}"

    if decision.reason == "opening":
        return Explanation(
            title,
            "On the first turn each player places a stone for the other side. "
            f"It put {them.possessive} stone in a corner, where it's least useful for "
            "building a road.",
        )
    if decision.reason == "win":
        return Explanation(title, _winning(after, me, them))
    if decision.reason == "lost":
        return Explanation(
            title,
            f"Whatever it played, {them.subject} could win next move, so it played the "
            "move its evaluation rated highest.",
        )
    if after.is_over:  # a move that ends the game without winning it: a draw
        return Explanation(title, f"{move_text} ends the game in a draw.")

    summary = _summary(before, decision, them)
    if decision.reason == "tinue":
        return Explanation(title, summary)
    points = _gains(agent, before, after, decision, them)
    return Explanation(title, summary, tuple(points), _outlook(decision))


def _winning(after: Game, me: Color, them: Opponent) -> str:
    result = after.result
    assert result is not None
    if result.reason is Reason.ROAD:
        return "It completes a road and wins the game."
    mine, theirs = count_flats(after.board, me), count_flats(after.board, me.opponent)
    return (
        f"It ends the game ahead on flats, {mine} to {theirs}, and wins. (The game ends "
        "when the board fills up or a player places their last piece.)"
    )


def _summary(before: Game, decision: Decision, them: Opponent) -> str:
    """The main reason for the move, in the order the learner's turn goes."""
    chosen = decision.chosen
    assert chosen is not None
    sentences = []
    threat = find_win(before, before.to_move.opponent)
    if threat is not None:
        only = " It was the only move that does." if decision.safe == 1 else ""
        sentences.append(
            f"{_cap(them.subject)} threatened to win with {format_move(threat)}, and this "
            f"move blocks it.{only}"
        )
    if decision.reason == "tinue":
        assert chosen.win_next is not None
        sentences.append(
            f"It makes Tinue: it threatens to win with {format_move(chosen.win_next)}, "
            f"and nothing {them.subject} can play stops it."
        )
        return " ".join(sentences)
    if threat is None or decision.safe > 1:
        choice = "ways to block" if threat is not None else "moves that don't hand"
        whom = "" if threat is not None else f" {them.subject} a win"
        among = f"Of the {decision.safe} {choice}{whom},"
        if not decision.passed_over:
            sentences.append(f"{among} this is the one its learned evaluation liked best.")
        else:
            option, reply = decision.passed_over[0]
            sentences.append(
                f"{among} its learned evaluation liked {format_move(option.move)} best, but "
                f"{them.subject} could have answered that with {format_move(reply)} for Tinue."
            )
            others = len(decision.passed_over) - 1
            if others:
                moves = "move" if others == 1 else "moves"
                sentences.append(f"The same was true of {others} more {moves} it liked.")
            sentences.append("So it played the best move that doesn't allow Tinue.")
    return " ".join(sentences)


# --- What the move gained --------------------------------------------------------------


def _gains(
    agent: LearnerAgent, before: Game, after: Game, decision: Decision, them: Opponent
) -> list[str]:
    """What the move gained, as the learned evaluation sees it, biggest first."""
    chosen = decision.chosen
    assert chosen is not None and chosen.features is not None
    me = before.to_move
    points = []
    if chosen.win_next is not None:
        points.append(
            f"Tak! It now threatens to win with {format_move(chosen.win_next)}, so "
            f"{them.subject} must block."
        )
    old = features(before, me, threat=False)
    new = chosen.features
    weight = dict(zip(FEATURE_NAMES, agent.weights.weights, strict=True))
    change = {
        name: weight[name] * (after_value - before_value)
        for name, before_value, after_value in zip(FEATURE_NAMES, old, new, strict=True)
    }
    gains = {group: sum(change[name] for name in names) for group, names in GROUPS.items()}
    ranked = sorted(
        (group for group, gain in gains.items() if gain >= MIN_GAIN),
        key=lambda group: gains[group],
        reverse=True,
    )
    facts = _Facts(before, after, me)
    for group in ranked:
        if len(points) == MAX_POINTS:
            break
        text = _DESCRIBE[group](facts, them)
        if text:
            points.append(text)
    if not points:
        points.append(
            "Nothing changes much: it's a quiet move that keeps its position as it is, "
            "which it judged better than the alternatives."
        )
    return points


class _Facts:
    """Plain facts about the board before and after the move, for describing gains."""

    def __init__(self, before: Game, after: Game, me: Color) -> None:
        self.size = before.size
        self.me = me
        self.them = me.opponent
        self.before = before
        self.after = after
        self.stacks_before = stacks_of(before.board)
        self.stacks_after = stacks_of(after.board)

    def flats(self, game: Game, color: Color) -> int:
        return count_flats(game.board, color)

    def gap(self, after: bool, color: Color) -> int:
        stacks = self.stacks_after if after else self.stacks_before
        return road_gap(stacks, self.size, color)

    def span(self, after: bool, color: Color) -> int:
        stacks = self.stacks_after if after else self.stacks_before
        return longest_road(stacks, self.size, color)

    def tops(self, after: bool, color: Color, kind: PieceType) -> int:
        stacks = self.stacks_after if after else self.stacks_before
        return sum(1 for s in stacks if s and s[-1].color is color and s[-1].type is kind)

    def held(self, after: bool, owner: Color, color: Color) -> int:
        """Stones of ``color`` buried in stacks that ``owner`` has on top."""
        stacks = self.stacks_after if after else self.stacks_before
        return sum(
            sum(1 for piece in s[:-1] if piece.color is color)
            for s in stacks
            if s and s[-1].color is owner
        )


def _squares(count: int) -> str:
    return f"{count} square" if count == 1 else f"{count} squares"


def _flats(facts: _Facts, them: Opponent) -> str | None:
    mine, theirs = facts.flats(facts.after, facts.me), facts.flats(facts.after, facts.them)
    lead = mine - theirs
    was = facts.flats(facts.before, facts.me) - facts.flats(facts.before, facts.them)
    if lead <= was:
        return None  # (the gain was only the end of the game drawing nearer)
    if lead > 0 and was <= 0:
        text = f"It takes the lead on flats, {mine} to {theirs}."
    elif lead > 0:
        text = f"It stretches its lead on flats: {mine} to {theirs} now."
    elif lead == 0:
        text = f"It draws level on flats, {mine} each."
    else:
        text = f"It closes the gap on flats to {mine} against {them.possessive} {theirs}."
    if endgame(facts.after) >= 0.3:
        text += " Near the end of the game, flats are what decide it."
    return text


def _roads(facts: _Facts, them: Opponent) -> str | None:
    blocked = facts.size * 3  # what road_gap says when no road is possible at all
    parts = []
    their_before, their_after = facts.gap(False, facts.them), facts.gap(True, facts.them)
    if their_after > their_before:
        if their_after >= blocked:
            parts.append(f"cuts off {them.possessive} last way to a road")
        else:
            parts.append(
                f"blocks {them.possessive} road ({them.verb('are', 'is')} now about "
                f"{_squares(their_after)} from one, up from {their_before})"
            )
    mine_before, mine_after = facts.gap(False, facts.me), facts.gap(True, facts.me)
    if mine_after < mine_before:
        parts.append(
            f"builds toward its own road (about {_squares(mine_after)} to go, "
            f"down from {mine_before})"
        )
    if not parts:
        span_before, span_after = facts.span(False, facts.me), facts.span(True, facts.me)
        if span_after > span_before:
            parts.append(f"stretches its longest connected line to {_squares(span_after)} across")
        else:
            parts.append("lines its pieces up better for a road")
    return "It " + " and ".join(parts) + "."


def _stacks(facts: _Facts, them: Opponent) -> str | None:
    prisoners = facts.held(True, facts.me, facts.them) - facts.held(False, facts.me, facts.them)
    captives = facts.held(True, facts.me, facts.me) - facts.held(False, facts.me, facts.me)
    if prisoners > 0:
        return (
            f"It buries {prisoners} of {them.possessive} stones under its own pieces, "
            f"where {them.subject} can't use {'it' if prisoners == 1 else 'them'}."
        )
    if captives > 0:
        return "It builds up a stack of its own stones that it can spread out later."
    return None


def _center(facts: _Facts, them: Opponent) -> str:
    return "It takes ground near the center, where a road can head in any direction."


def _walls(facts: _Facts, them: Opponent) -> str | None:
    if facts.tops(True, facts.them, PieceType.WALL) < facts.tops(False, facts.them, PieceType.WALL):
        return f"It flattens {them.possessive} wall with its capstone."
    if facts.tops(True, facts.me, PieceType.WALL) > facts.tops(False, facts.me, PieceType.WALL):
        return "It stands up a wall, which blocks roads and can't be stacked on."
    return None


def _capstone(facts: _Facts, them: Opponent) -> str | None:
    if facts.tops(True, facts.me, PieceType.CAP) > facts.tops(False, facts.me, PieceType.CAP):
        return "It brings out its capstone, which can't be covered and can flatten walls."
    return None


def _reserves(facts: _Facts, them: Opponent) -> str:
    return "It puts another piece into play, which it learned to value by playing itself."


_DESCRIBE = {
    "flats": _flats,
    "roads": _roads,
    "stacks": _stacks,
    "center": _center,
    "walls": _walls,
    "capstone": _capstone,
    "reserves": _reserves,
}


def _outlook(decision: Decision) -> str | None:
    """How the learner rates its position after the move, in words."""
    chosen = decision.chosen
    if chosen is None or chosen.features is None:
        return None
    rating = chosen.value
    if rating > 0.6:
        words = "well ahead"
    elif rating > 0.2:
        words = "ahead"
    elif rating >= -0.2:
        words = "about even"
    elif rating >= -0.6:
        words = "behind"
    else:
        words = "well behind"
    return f"How it rates its position now: {words} ({rating:+.2f}, on a scale of -1 to +1)."


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]
