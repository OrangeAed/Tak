"""The learner: its features, its learned evaluation, its endgame search, and its training."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from tak.ai import LearnerAgent, make_agent
from tak.ai.learner_agent import allows_tinue, find_win
from tak.ai.learning import (
    FEATURE_NAMES,
    NUM_FEATURES,
    WEIGHTS_PATH,
    Weights,
    default_weights,
    features,
    load_weights,
    save_weights,
    value,
)
from tak.ai.tactics import find_win as tactics_find_win
from tak.ai.tactics import flat_win_possible
from tak.ai.training import Settings, Trajectory, self_play, td_update, train
from tak.engine import Color, Game, Move, PieceType, PlaceMove, game_from_ptn, parse_move

from .conftest import Position, bot_game_positions, random_positions
from .test_agents import BLACK_TINUE, WHITE_TINUE


def wins_for_mover(game: Game, move: Move) -> bool:
    """Whether ``move`` wins for the player to move, by really playing it."""
    after = game.copy()
    mover = after.to_move
    after.play(move)
    return after.result is not None and after.result.winner is mover


def allows_tinue_brute_force(game: Game, move: Move) -> bool:
    """Whether the opponent could reply to ``move`` with Tinue, using the engine."""
    after = game.copy()
    them = after.to_move.opponent
    after.play(move)
    for reply in after.legal_moves():
        after.play(reply)
        try:
            if after.result is None and after.is_tinue(them):
                return True
        finally:
            after.undo()
    return False


def endgame_positions(size: int, count: int, seed: int) -> list[Game]:
    """Positions near the end of random games, where flat wins are possible."""
    rng = random.Random(seed)
    found: list[Game] = []
    while len(found) < count:
        game = Game(size)
        while not game.is_over and len(found) < count:
            game.play(rng.choice(game.legal_moves()))
            if not game.is_over and not game.in_opening and flat_win_possible(game, Color.WHITE):
                found.append(game.copy())
    return found


class TestWeights:
    def test_the_weights_that_come_with_the_game_load(self) -> None:
        weights = load_weights()
        assert len(weights.weights) == NUM_FEATURES
        assert weights.games > 0  # they were trained
        assert any(weight != 0 for weight in weights.weights)
        assert default_weights() == weights

    def test_save_and_load(self, tmp_path: Path) -> None:
        weights = Weights([0.5 * i for i in range(NUM_FEATURES)], -0.25, games=7, sizes=(4, 6))
        save_weights(weights, tmp_path / "w.json")
        assert load_weights(tmp_path / "w.json") == weights

    def test_the_file_names_every_feature(self) -> None:
        text = WEIGHTS_PATH.read_text(encoding="utf-8")
        for name in FEATURE_NAMES:
            assert f'"{name}"' in text

    def test_a_file_missing_a_feature_is_refused(self) -> None:
        text = Weights.zeros().to_json().replace('"tak"', '"something_else"')
        with pytest.raises(ValueError, match="missing features: tak"):
            Weights.from_json(text)

    def test_untrained_weights_see_every_position_as_even(self) -> None:
        assert value(Weights.zeros(), [1.0] * NUM_FEATURES) == 0.0

    def test_values_are_between_minus_one_and_one(self) -> None:
        weights = Weights([100.0] * NUM_FEATURES)
        assert 0.99 < value(weights, [1.0] * NUM_FEATURES) <= 1.0
        assert -1.0 <= value(weights, [-1.0] * NUM_FEATURES) < -0.99


class TestFeatures:
    @pytest.mark.parametrize("size", [3, 4, 5, 6, 7, 8])
    def test_one_number_per_feature(self, size: int) -> None:
        for game in random_positions(size, 3, seed=size):
            assert len(features(game, Color.WHITE, threat=False)) == NUM_FEATURES

    def test_an_empty_board_is_even(self) -> None:
        game = Game(5)
        x = features(game, Color.WHITE, threat=False)
        assert x == [0.0] * NUM_FEATURES

    def test_ours_minus_theirs(self) -> None:
        # Swapping whose point of view it is flips the sign of every "ours
        # minus theirs" feature, swaps "our" and "their" road gaps, and leaves
        # "tak" (about the player who just moved) alone.
        names = list(FEATURE_NAMES)
        swapped = {"our_gap_2": "their_gap_2", "our_gap_3": "their_gap_3"}
        swapped.update({theirs: ours for ours, theirs in swapped.items()})
        for game in random_positions(5, 10, seed=3):
            white = dict(zip(names, features(game, Color.WHITE, threat=True), strict=True))
            black = dict(zip(names, features(game, Color.BLACK, threat=True), strict=True))
            for name in names:
                if name == "tak":
                    assert white[name] == black[name] == 1.0
                elif name in swapped:
                    assert white[name] == black[swapped[name]]
                else:
                    assert white[name] == pytest.approx(-black[name])

    def test_what_it_measures(self, position: Position) -> None:
        game = position(
            5,
            {"a1": "w", "b1": "w", "c1": "w", "d1": "w", "a5": "b", "c3": "Sb", "e5": "b w"},
            white=(15, 1),
            black=(17, 1),
        )
        x = dict(zip(FEATURE_NAMES, features(game, Color.WHITE, threat=True), strict=True))
        assert x["flats"] == pytest.approx((5 - 1) / 5)  # a1-d1 and e5 against a5
        assert x["walls"] == pytest.approx(-1 / 5)
        assert x["prisoners"] == pytest.approx(1 / 5)  # Black's stone under e5
        assert x["road_squares"] == 1.0  # e1 completes White's road
        assert x["road_gap"] > 0
        assert x["longest_road"] > 0
        assert x["tak"] == 1.0
        assert x["reserves"] < 0  # White has placed more
        assert x["our_gap_2"] == x["our_gap_3"] == 0.0  # White is one short, not two or three


class TestEndgameWins:
    """The learner's own win search, checked against the engine's brute force."""

    @pytest.mark.parametrize("size", [3, 4, 5, 6])
    def test_agrees_with_the_engine_near_the_end(self, size: int) -> None:
        positions = endgame_positions(size, 40, seed=size)
        found = 0
        for game in positions:
            for color in Color:
                fast = find_win(game, color)
                assert (fast is None) == (game.winning_move(color) is None)
                if fast is not None:
                    found += 1
                    if color is game.to_move:
                        assert wins_for_mover(game, fast)
        assert found > 0

    @pytest.mark.parametrize("size", [4, 5])
    def test_agrees_in_games_between_computer_players(self, size: int) -> None:
        for game in bot_game_positions(size, seed=size, white="strategist", black="greedy"):
            for color in Color:
                fast = find_win(game, color)
                assert (fast is None) == (tactics_find_win(game, color) is None)

    def test_a_flat_win_by_placing_the_last_piece(self, position: Position) -> None:
        game = position(4, {"a1": "w", "b2": "w", "c3": "b"}, white=(1, 0))
        move = find_win(game, Color.WHITE)
        assert isinstance(move, PlaceMove) and move.piece is PieceType.FLAT

    def test_placing_the_last_piece_when_behind_is_no_win(self, position: Position) -> None:
        game = position(4, {"a1": "b", "b2": "b", "c3": "w"}, white=(1, 0))
        assert find_win(game, Color.WHITE) is None
        assert game.winning_move(Color.WHITE) is None


@pytest.fixture(scope="module")
def contested() -> list[Game]:
    """Positions from games of the strategist against the learner, full of threats.

    Only positions where the player to move can't win at once. Built once for
    the module; tests must leave them as they found them.
    """
    games = bot_game_positions(4, seed=3, white="strategist", black="learner")
    games += bot_game_positions(4, seed=4, white="learner", black="strategist")
    return [game for game in games if game.winning_move(game.to_move) is None]


class TestLearnerAgent:
    def test_registered(self) -> None:
        agent = make_agent("learner", seed=1)
        assert isinstance(agent, LearnerAgent)
        assert agent.name == "Learner"
        assert agent.weights == default_weights()

    @pytest.mark.parametrize(("ptn", "tinue"), [(WHITE_TINUE, "d3<"), (BLACK_TINUE, "d3+")])
    def test_plays_the_move_that_makes_tinue(self, ptn: str, tinue: str) -> None:
        game = game_from_ptn(ptn)
        for seed in range(3):
            assert LearnerAgent(seed).choose_move(game) == parse_move(tinue)

    def test_without_tinue_checks_it_plays_the_safe_move_it_likes_best(
        self, contested: list[Game]
    ) -> None:
        agent = LearnerAgent(seed=0, tinue_checks=0)
        checked = 0
        for game in contested:
            options = agent.consider(game.copy(), random.Random(0))
            if any(option.win_next is not None for option in options if option.safe):
                continue  # it might be Tinue, which comes first
            safe = {option.move: option.value for option in options if option.safe}
            if safe:
                chosen = agent.choose_move(game)
                assert safe[chosen] == pytest.approx(max(safe.values()))
                checked += 1
        assert checked >= 5

    def test_passes_over_moves_that_let_the_opponent_make_tinue(
        self, contested: list[Game]
    ) -> None:
        agent = LearnerAgent(seed=0)
        passed_over = 0
        for game in contested:
            options = agent.consider(game.copy(), random.Random(0))
            if any(option.win_next is not None for option in options if option.safe):
                continue
            ranked = sorted((o for o in options if o.safe), key=lambda o: o.value, reverse=True)
            if not ranked:
                continue
            # (allows_tinue itself is checked against the engine below.)
            checked = ranked[: agent.tinue_checks]
            fine = [o for o in checked if not allows_tinue(game, o.move)]
            expected = fine[0] if fine else ranked[0]
            chosen = agent.choose_move(game)
            assert expected.value == pytest.approx(
                next(o.value for o in options if o.move == chosen)
            )
            if fine and fine[0] is not ranked[0]:
                passed_over += 1
        assert passed_over >= 1

    def test_allows_tinue_agrees_with_the_engine(self, contested: list[Game]) -> None:
        found = 0
        for game in contested[::4]:
            moves = [m for m in game.legal_moves() if not self.unsafe(game, m)][::6]
            for move in moves:
                expected = allows_tinue_brute_force(game, move)
                assert allows_tinue(game, move) == expected
                found += expected
        assert found >= 1

    @staticmethod
    def unsafe(game: Game, move: Move) -> bool:
        after = game.copy()
        mover = after.to_move
        after.play(move)
        if after.result is not None:
            return True
        return after.winning_move(mover.opponent) is not None

    def test_different_weights_play_differently(self) -> None:
        # Only flats matter to one, only walls to the other.
        flats, walls = Weights.zeros(), Weights.zeros()
        flats.weights[FEATURE_NAMES.index("flats")] = 1.0
        walls.weights[FEATURE_NAMES.index("walls")] = 1.0
        game = Game(5)
        for move in ("a1", "e5"):
            game.play(parse_move(move))
        by_flats = LearnerAgent(seed=0, weights=flats).choose_move(game)
        by_walls = LearnerAgent(seed=0, weights=walls).choose_move(game)
        assert isinstance(by_flats, PlaceMove) and by_flats.piece is PieceType.FLAT
        assert isinstance(by_walls, PlaceMove) and by_walls.piece is PieceType.WALL

    def test_the_trained_weights_beat_the_untrained_ones(self) -> None:
        score = 0
        for number in range(4):
            trained = LearnerAgent(seed=number)
            untrained = LearnerAgent(seed=number, weights=Weights.zeros())
            trained_color = Color.WHITE if number % 2 == 0 else Color.BLACK
            game = Game(4)
            while not game.is_over:
                agent = trained if game.to_move is trained_color else untrained
                game.play(agent.choose_move(game))
            assert game.result is not None
            score += game.result.winner is trained_color
        assert score >= 3


class TestTraining:
    SMALL = Settings(games=4, sizes=(3,), batch=2, seed=5)

    def test_self_play_plays_a_whole_game(self) -> None:
        game = self_play(Weights.zeros(), self.SMALL, 0)
        assert game.positions
        assert all(len(x) == NUM_FEATURES for x, _ in game.positions)
        # The players take turns, so the point of view alternates.
        whites = [white for _, white in game.positions]
        assert all(a != b for a, b in zip(whites, whites[1:], strict=False))

    def test_self_play_is_repeatable(self) -> None:
        one = self_play(Weights.zeros(), self.SMALL, 3)
        two = self_play(Weights.zeros(), self.SMALL, 3)
        assert one == two

    def test_a_won_game_raises_the_winners_positions(self) -> None:
        x = [1.0] + [0.0] * (NUM_FEATURES - 1)
        minus = [-v for v in x]
        # White moved into positions with x (from its point of view), Black into -x, White won.
        game = Trajectory([(x, True), (minus, False), (x, True)], Color.WHITE, 5)
        weights = Weights.zeros()
        td_update(weights, game, learning_rate=0.1, lam=0.7)
        assert weights.weights[0] > 0
        assert value(weights, x) > 0
        assert weights.games == 1

    def test_a_draw_teaches_nothing_from_nothing(self) -> None:
        x = [0.5] * NUM_FEATURES
        game = Trajectory([(x, True), (x, False)], None, 4)
        weights = Weights.zeros()
        assert td_update(weights, game, learning_rate=0.1, lam=0.7) == 0.0
        assert weights.weights == [0.0] * NUM_FEATURES

    def test_training_changes_the_weights_and_reports_progress(self) -> None:
        seen = []
        weights = train(self.SMALL, workers=1, report=seen.append)
        assert weights.games == 4
        assert weights.sizes == (3,)
        assert weights.weights != [0.0] * NUM_FEATURES
        assert [progress.games for progress in seen] == [2, 4]

    def test_the_same_seed_trains_the_same_weights_on_any_number_of_processes(self) -> None:
        assert train(self.SMALL, workers=1) == train(self.SMALL, workers=2)

    def test_training_on_several_sizes(self) -> None:
        weights = train(Settings(games=4, sizes=(3, 4), batch=2, seed=1), workers=1)
        assert weights.games == 4
        assert weights.sizes == (3, 4)

    def test_a_time_limit_stops_training_early(self) -> None:
        weights = train(Settings(games=100, sizes=(3,), batch=2), workers=1, time_limit=0)
        assert weights.games == 2
