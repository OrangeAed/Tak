"""The evaluation bar's reading: who is ahead, and by how much, as the learner sees it."""

from __future__ import annotations

import pytest

from tak.ai import LearnerAgent
from tak.ai.assess import FAR, NEAR, _words, assess
from tak.ai.learning import Weights
from tak.engine import Color, Game, game_from_ptn, parse_move

from .conftest import Position, bot_game_positions
from .test_agents import WHITE_TINUE


def after(*moves: str, size: int = 5) -> Game:
    game = Game(size)
    for move in moves:
        game.play(parse_move(move))
    return game


class TestReadings:
    def test_the_start_is_about_even_with_white_to_move_a_little_better(self) -> None:
        reading = assess(Game(5))
        assert 0 < reading.white <= NEAR
        assert reading.leader is Color.WHITE
        assert reading.detail.startswith("It's about even (+0.")
        assert reading.label == f"{reading.white:.2f}" and not reading.decided

    def test_untrained_weights_see_nothing_in_it(self) -> None:
        reading = assess(after("a1", "e5", "c3"), Weights.zeros())
        assert reading.white == 0.0 and reading.leader is None

    def test_a_finished_game_shows_its_result(self) -> None:
        won = after("a1", "c3", "b1", "b3", "b2", "a3", "c2", size=3)  # White's road
        reading = assess(won)
        assert (reading.white, reading.label, reading.decided) == (1.0, "R-0", True)
        assert reading.detail == "White has won (R-0)."

    def test_a_draw(self, position: Position) -> None:
        game = position(3, {"a1": "w", "b2": "b", "c1": "b"}, white=(1, 0))
        game.play(parse_move("c3"))  # White's last piece, leaving the flats level at 2-2
        assert game.result is not None and game.result.winner is None
        reading = assess(game)
        assert (reading.white, reading.label, reading.decided) == (0.0, "Draw", True)

    def test_the_player_to_move_can_win_now(self, position: Position) -> None:
        game = position(4, {"a2": "b", "b2": "b", "c2": "b"}, to_move=Color.BLACK)
        reading = assess(game)
        assert (reading.white, reading.label, reading.decided) == (-1.0, "Win", True)
        assert reading.detail == "Black can win with their next move."

    def test_tinue(self) -> None:
        game = game_from_ptn(WHITE_TINUE)
        game.play(parse_move("d3<"))  # the move that makes Tinue
        reading = assess(game)
        assert (reading.white, reading.label, reading.decided) == (1.0, "Win", True)
        assert reading.detail == "White has Tinue: they win whatever Black plays."

    def test_a_threat_that_can_be_stopped_is_not_decided(self, position: Position) -> None:
        # White threatens d2, but Black can block it.
        game = position(4, {"a2": "w", "b2": "w", "c2": "w", "a4": "b"}, to_move=Color.BLACK)
        reading = assess(game)
        assert not reading.decided and -1 < reading.white < 1

    @pytest.mark.parametrize("white", [0.75, 0.35, 0.05, -0.35, -0.75])
    def test_words_follow_the_numbers(self, white: float) -> None:
        text = _words(white)
        if abs(white) <= NEAR:
            assert text == "It's about even"
        else:
            leader = "White" if white > 0 else "Black"
            assert text == f"{leader} is {'well ahead' if abs(white) > FAR else 'ahead'}"


class TestAgreesWithTheLearner:
    def test_the_bar_shows_what_the_learner_says_after_its_move(self) -> None:
        agent = LearnerAgent(seed=1)
        checked = 0
        games = bot_game_positions(5, seed=3, white="learner", black="strategist")
        games += bot_game_positions(4, seed=4, white="strategist", black="learner")
        for game in games:
            decision = agent.decide(game)
            chosen = decision.chosen
            if decision.reason != "best" or chosen is None or chosen.features is None:
                continue
            later = game.copy()
            later.play(decision.move)
            reading = assess(later)
            if reading.decided:
                continue
            mine = reading.white if game.to_move is Color.WHITE else -reading.white
            assert mine == pytest.approx(chosen.value)
            checked += 1
        assert checked >= 10

    def test_leaves_the_game_alone(self) -> None:
        for game in bot_game_positions(4, seed=5, white="learner", black="greedy")[::3]:
            before = ([game.board.stack(s) for s in game.board.squares()], game.ply)
            assess(game)
            assert ([game.board.stack(s) for s in game.board.squares()], game.ply) == before
