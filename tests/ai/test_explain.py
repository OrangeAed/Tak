"""Explaining the learner's moves: what it says, and that it's true to what the learner did."""

from __future__ import annotations

import re

import pytest

from tak.ai import LearnerAgent, make_agent
from tak.ai.evaluation import road_gap
from tak.ai.explain import Explanation, can_explain, explain_move
from tak.ai.learner_agent import tinue_reply
from tak.ai.tactics import stacks_of
from tak.engine import Game, format_move, game_from_ptn, parse_move
from tak.engine.rules import count_flats

from .conftest import Position, bot_game_positions
from .test_agents import WHITE_TINUE


def everything(explanation: Explanation) -> str:
    parts = [explanation.title, explanation.summary, *explanation.points]
    return " ".join(parts + ([explanation.outlook] if explanation.outlook else []))


def played_text(game: Game, agent: LearnerAgent) -> str:
    """The learner's move in ``game``, written as it is once played."""
    after = game.copy()
    after.play(agent.choose_move(game))
    played = after.history[-1]
    return format_move(played.move, flattened=played.flattened)


@pytest.fixture(scope="module")
def agent() -> LearnerAgent:
    """One learner for the module, so each position is only worked out once."""
    return LearnerAgent(seed=1)


@pytest.fixture(scope="module")
def positions() -> list[Game]:
    """Positions from games between the learner and the strategist, both ways round."""
    games = bot_game_positions(5, seed=1, white="learner", black="strategist")
    games += bot_game_positions(5, seed=2, white="strategist", black="learner")
    games += bot_game_positions(4, seed=3, white="strategist", black="learner")
    games += bot_game_positions(4, seed=4, white="learner", black="strategist")
    return games


class TestWhatItSays:
    def test_the_opening(self) -> None:
        agent = LearnerAgent(seed=1)
        explanation = explain_move(agent, Game(5))
        assert explanation.title == f"Why Learner played {played_text(Game(5), agent)}"
        assert "corner" in explanation.summary
        assert explanation.points == () and explanation.outlook is None

    def test_a_road_win(self, position: Position) -> None:
        game = position(5, {"a3": "w", "b3": "w", "c3": "w", "d3": "w", "a1": "b", "b1": "b"})
        explanation = explain_move(LearnerAgent(seed=1), game)
        assert explanation.summary == "It completes a road and wins the game."

    def test_a_flat_win(self, position: Position) -> None:
        game = position(4, {"a1": "w", "b2": "w", "c3": "b"}, white=(1, 0))
        explanation = explain_move(LearnerAgent(seed=1), game)
        assert explanation.summary.startswith("It ends the game ahead on flats, 3 to 1")

    def test_a_block(self, position: Position) -> None:
        # Black threatens d2, and White can only stop it there (a flat or a wall).
        game = position(4, {"a2": "b", "b2": "b", "c2": "b", "a4": "w", "c4": "w"})
        explanation = explain_move(LearnerAgent(seed=1), game)
        assert explanation.summary.startswith(
            "You threatened to win with d2, and this move blocks it. Of the 2 ways to block,"
        )

    def test_tinue(self) -> None:
        game = game_from_ptn(WHITE_TINUE)
        explanation = explain_move(LearnerAgent(seed=1), game)
        assert explanation.title == "Why Learner played d3<"
        assert "It makes Tinue: it threatens to win with" in explanation.summary
        assert "nothing you can play stops it" in explanation.summary

    def test_a_quiet_move(self, agent: LearnerAgent, positions: list[Game]) -> None:
        quiet = [
            game
            for game in positions
            if agent.decide(game).reason == "best"
            and not agent.decide(game).passed_over
            and "threatened" not in explain_move(agent, game).summary
        ]
        assert len(quiet) >= 10
        for game in quiet:
            explanation = explain_move(agent, game)
            assert "this is the one its learned evaluation liked best" in explanation.summary
            assert 1 <= len(explanation.points) <= 3
            assert explanation.outlook is not None
            assert explanation.outlook.startswith("How it rates its position now:")

    def test_a_threat_comes_first(self, agent: LearnerAgent, positions: list[Game]) -> None:
        taks = []
        for game in positions:
            decision = agent.decide(game)
            if decision.reason == "best" and decision.chosen and decision.chosen.win_next:
                taks.append(game)
        assert taks
        for game in taks:
            assert explain_move(agent, game).points[0].startswith("Tak! It now threatens to win")

    def test_a_move_passed_over_for_allowing_tinue(
        self, agent: LearnerAgent, positions: list[Game]
    ) -> None:
        passed = [game for game in positions if agent.decide(game).passed_over]
        assert passed
        for game in passed:
            option, reply = agent.decide(game).passed_over[0]
            summary = explain_move(agent, game).summary
            assert (
                f"its learned evaluation liked {format_move(option.move)} best, but you could "
                f"have answered that with {format_move(reply)} for Tinue." in summary
            )
            assert summary.endswith("So it played the best move that doesn't allow Tinue.")

    def test_a_lost_position(self, agent: LearnerAgent, positions: list[Game]) -> None:
        lost = [game for game in positions if agent.decide(game).reason == "lost"]
        assert lost
        for game in lost:
            summary = explain_move(agent, game).summary
            assert summary.startswith("Whatever it played, you could win next move")

    def test_against_another_computer_the_opponent_is_named_by_color(
        self, agent: LearnerAgent, positions: list[Game]
    ) -> None:
        for game in positions[::3]:
            text = everything(explain_move(agent, game, person=False))
            assert not re.search(r"\byou(r)?\b", text, re.IGNORECASE), text
        game = game_from_ptn(WHITE_TINUE)
        assert "nothing Black can play stops it" in explain_move(agent, game, person=False).summary


class TestTrueToTheMove:
    def test_explains_the_move_it_plays(self, positions: list[Game]) -> None:
        for game in positions[::6]:
            explanation = explain_move(LearnerAgent(seed=7), game)
            other = LearnerAgent(seed=7)  # nothing remembered: works it out afresh
            assert explanation.title == f"Why Learner played {played_text(game, other)}"

    def test_leaves_the_game_alone(self, agent: LearnerAgent, positions: list[Game]) -> None:
        for game in positions[::5]:
            before = ([game.board.stack(s) for s in game.board.squares()], game.ply)
            explain_move(agent, game)
            assert ([game.board.stack(s) for s in game.board.squares()], game.ply) == before

    def test_gains_it_claims_really_happened(
        self, agent: LearnerAgent, positions: list[Game]
    ) -> None:
        claims = 0
        for game in positions:
            if agent.decide(game).reason != "best":
                continue
            me, them = game.to_move, game.to_move.opponent
            after = game.copy()
            after.play(agent.choose_move(game))
            for point in explain_move(agent, game).points:
                if "on flats" in point:
                    lead = count_flats(after.board, me) - count_flats(after.board, them)
                    was = count_flats(game.board, me) - count_flats(game.board, them)
                    assert lead > was, point
                    claims += 1
                if "blocks your road" in point:
                    size = game.size
                    was_gap = road_gap(stacks_of(game.board), size, them)
                    assert road_gap(stacks_of(after.board), size, them) > was_gap, point
                    claims += 1
        assert claims >= 10


class TestDeciding:
    def test_decide_is_choose_move_with_its_reasons(self, positions: list[Game]) -> None:
        for game in positions[::3]:
            decision = LearnerAgent(seed=3).decide(game)
            assert decision.move == LearnerAgent(seed=3).choose_move(game)

    def test_decisions_are_remembered_and_forgotten_in_time(self, positions: list[Game]) -> None:
        agent = LearnerAgent(seed=3)
        agent.remembered = 2
        first = agent.decide(positions[10])
        assert agent.decide(positions[10]) is first  # remembered
        agent.decide(positions[11])
        agent.decide(positions[12])
        assert agent.decide(positions[10]) is not first  # forgotten, so worked out again
        assert agent.decide(positions[10]) == first

    def test_the_tinue_reply_really_makes_tinue(
        self, agent: LearnerAgent, positions: list[Game]
    ) -> None:
        found = 0
        for game in positions:
            for option, reply in agent.decide(game).passed_over:
                after = game.copy()
                after.play(option.move)
                after.play(reply)
                assert after.is_tinue(game.to_move.opponent)
                assert tinue_reply(game, option.move) == reply
                found += 1
        assert found >= 1

    def test_only_the_learner_explains(self) -> None:
        assert can_explain(LearnerAgent())
        assert not can_explain(make_agent("strategist"))
        assert not can_explain(None)

    def test_a_finished_game_has_nothing_to_explain(self) -> None:
        game = Game(3)
        for move in ["a1", "c3", "b1", "b3", "b2", "a3", "c2"]:
            game.play(parse_move(move))
        with pytest.raises(ValueError, match="over"):
            explain_move(LearnerAgent(), game)
