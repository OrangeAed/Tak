"""The position measurements the strategist maximizes."""

from __future__ import annotations

import math

import pytest

from tak.ai.evaluation import (
    centrality,
    endgame,
    evaluate,
    features,
    longest_road,
    road_gap,
)
from tak.ai.tactics import stacks_of
from tak.engine import Color, Game, parse_move

from .conftest import Position, random_positions


class TestLongestRoad:
    def test_an_empty_board_has_no_road(self) -> None:
        game = Game(5)
        assert longest_road(stacks_of(game.board), 5, Color.WHITE) == 0

    def test_counts_the_widest_reach_of_a_connected_group(self, position: Position) -> None:
        game = position(5, {"a1": "w", "b1": "w", "b2": "w", "c2": "w", "e5": "w", "a3": "b"})
        stacks = stacks_of(game.board)
        assert longest_road(stacks, 5, Color.WHITE) == 3  # a1-b1-b2-c2 spans files a to c
        assert longest_road(stacks, 5, Color.BLACK) == 1

    def test_walls_and_covered_stones_dont_count(self, position: Position) -> None:
        game = position(4, {"a1": "w", "b1": "Sw", "c1": "w b", "d1": "w"})
        assert longest_road(stacks_of(game.board), 4, Color.WHITE) == 1


class TestRoadGap:
    def test_an_empty_board_needs_a_whole_row(self) -> None:
        assert road_gap(stacks_of(Game(6).board), 6, Color.WHITE) == 6

    def test_own_stones_count_as_done(self, position: Position) -> None:
        game = position(4, {"a2": "w", "b2": "w", "c2": "w"})
        assert road_gap(stacks_of(game.board), 4, Color.WHITE) == 1

    def test_opponents_stones_cost_more_and_capstones_block(self, position: Position) -> None:
        open_row = position(3, {"a1": "w", "b1": "b", "c1": "w", "b2": "b", "b3": "b"})
        assert road_gap(stacks_of(open_row.board), 3, Color.WHITE) == 2  # capture b1
        # A cross of capstones cuts every road in both directions.
        cross = {name: "Cb" for name in ("b1", "b2", "b3", "a2", "c2")}
        walled = position(3, cross)
        assert road_gap(stacks_of(walled.board), 3, Color.WHITE) == 9  # no way through


def test_centrality_is_highest_in_the_middle() -> None:
    weights = centrality(5)
    assert weights[12] == 1.0  # c3
    assert weights[0] == weights[4] == weights[20] == weights[24] == 0.0
    assert 0.0 < weights[7] < 1.0  # c2
    assert centrality(3)[4] == 1.0


def test_features_count_squares_flats_and_captives(position: Position) -> None:
    game = position(4, {"a1": "w", "b1": "Sw", "c1": "b w", "d1": "w w", "d4": "Cb"})
    white = features(game.board, Color.WHITE)
    assert white.control == 4
    assert white.flats == 3  # the wall isn't a flat
    assert white.captives == 1  # d1's lower stone; the black stone under c1 isn't White's
    black = features(game.board, Color.BLACK)
    assert (black.control, black.flats) == (1, 0)


class TestEvaluate:
    @pytest.mark.parametrize("size", [4, 5, 6])
    def test_one_players_gain_is_the_others_loss(self, size: int) -> None:
        for game in random_positions(size, 10, seed=size):
            assert evaluate(game, Color.WHITE) == pytest.approx(-evaluate(game, Color.BLACK))

    def test_finished_games(self) -> None:
        game = Game(3)
        for token in ["a1", "a2", "b2", "a3", "c2"]:
            game.play(parse_move(token))
        assert evaluate(game, Color.WHITE) == math.inf
        assert evaluate(game, Color.BLACK) == -math.inf

    def test_a_longer_road_is_better(self, position: Position) -> None:
        ahead = position(5, {"a3": "w", "b3": "w", "c3": "w", "e5": "b", "e1": "b"})
        level = position(5, {"a3": "w", "c5": "w", "c1": "w", "e5": "b", "e1": "b"})
        assert evaluate(ahead, Color.WHITE) > evaluate(level, Color.WHITE)

    def test_the_endgame_weighs_more_as_the_board_fills(self, position: Position) -> None:
        assert endgame(Game(5)) == 0.0
        nearly_full = position(
            3, {name: "w" for name in ["a1", "b1", "c1", "a2", "b2", "c2", "a3", "b3"]}
        )
        assert endgame(nearly_full) > 0.5
