"""Tests for spotting a threat to win next turn ("Tak").

A player threatens to win if, were they to move again right away, some legal
move would win them the game by any win condition.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from tak.engine import Color, Direction, Game, Move, PieceType, PlaceMove, SlideMove, Square

Position = Callable[..., Game]
Started = Callable[..., Game]
Snapshot = Callable[[Game], Any]
WinsWhenPlayed = Callable[[Game, Color, Move], bool]
SwapColors = Callable[[dict[str, str]], dict[str, str]]

TAK_GAME = "a1 c3 b1 b3 b2"  # 3x3: White's b2 threatens c2, joining b1-b2-c2-c3


def sq(name: str) -> Square:
    return Square.from_name(name)


class TestRoadThreats:
    def test_one_placement_from_a_road(self, started: Started) -> None:
        game = started(TAK_GAME, size=3)
        assert game.to_move is Color.BLACK
        assert game.threatens_win(Color.WHITE)
        assert game.winning_move(Color.WHITE) == PlaceMove(sq("c2"))

    def test_the_threat_is_judged_for_the_player_asked_about(self, started: Started) -> None:
        game = started(TAK_GAME, size=3)
        assert not game.threatens_win(Color.BLACK)  # Black has only a1 and b3

    def test_no_threat_at_the_start(self, started: Started) -> None:
        for moves in ["", "a1", "a1 c3", "a1 c3 b1"]:
            game = started(moves, size=3)
            assert not game.threatens_win(Color.WHITE)
            assert not game.threatens_win(Color.BLACK)

    def test_two_placements_short_is_not_a_threat(self, position: Position) -> None:
        game = position(5, {"a1": "w", "b1": "w", "c1": "w"})
        assert not game.threatens_win(Color.WHITE)

    def test_a_wall_in_the_way_blocks_the_threat(self, position: Position) -> None:
        game = position(3, {"a1": "w", "b1": "Sb", "c1": "w", "b2": "b", "b3": "b"})
        assert not game.threatens_win(Color.WHITE)

    def test_a_slide_that_spreads_a_stack_into_a_road(self, position: Position) -> None:
        # Only spreading a2 across b2 and c2 finishes a road in one move.
        game = position(3, {"a2": "w w w"})
        assert game.winning_move(Color.WHITE) == SlideMove(sq("a2"), Direction.RIGHT, (1, 1))

    def test_a_capstone_flattening_a_wall_completes_the_road(self, position: Position) -> None:
        stacks = {"a1": "w", "b1": "w", "c1": "Sb", "d1": "w", "e1": "w", "c2": "Cw"}
        game = position(5, stacks)
        assert game.winning_move(Color.WHITE) == SlideMove(sq("c2"), Direction.DOWN, (1,))

    def test_placing_a_capstone_when_out_of_stones(self, position: Position) -> None:
        stacks = {"a1": "w", "b1": "w", "c1": "w", "d1": "w", "a3": "b"}
        game = position(5, stacks, white=(0, 1))
        # No stones are left, so the road has to be finished with the capstone.
        assert game.winning_move(Color.WHITE) == PlaceMove(sq("e1"), PieceType.CAP)

    def test_completing_only_the_opponents_road_is_not_a_win(self, position: Position) -> None:
        # White can uncover c1 to finish Black's road, but White has no road to make.
        stacks = {"a1": "b", "b1": "b", "c1": "b w", "a3": "Sb", "b3": "Sb"}
        assert not position(3, stacks).threatens_win(Color.WHITE)

    def test_completing_both_roads_is_a_win_for_the_mover(self, position: Position) -> None:
        # 2c1+11 finishes White's top row but also uncovers Black's bottom row.
        # The mover wins a double road, and no other move wins, so this is the threat.
        stacks = {"a1": "b", "b1": "b", "c1": "b w w", "a3": "w", "b3": "w", "c3": "b"}
        game = position(3, stacks)
        assert game.winning_move(Color.WHITE) == SlideMove(sq("c1"), Direction.UP, (1, 1))


class TestFlatThreats:
    # White leads on flats, 2 to 1, with nothing close to a road.
    STACKS = {"a1": "w", "c1": "w", "b2": "b"}

    def test_placing_the_last_piece_while_ahead(self, position: Position) -> None:
        assert position(3, self.STACKS, white=(1, 0)).threatens_win(Color.WHITE)

    def test_placing_the_last_piece_while_behind(self, position: Position) -> None:
        # Black's last flat would only tie it 2 to 2: a draw is not a win.
        game = position(3, self.STACKS, black=(1, 0))
        assert not game.threatens_win(Color.BLACK)

    def test_placing_the_last_piece_to_go_ahead(self, position: Position) -> None:
        # With c3 too, Black's last flat makes it 3 to 2 for Black.
        game = position(3, {**self.STACKS, "c3": "b"}, black=(1, 0))
        assert game.threatens_win(Color.BLACK)

    def test_filling_the_board_while_ahead(self, position: Position) -> None:
        stacks = {
            "a1": "w", "b1": "Sb", "c1": "w",
            "a2": "Sb", "b2": "w", "c2": "Sb",
            "a3": "b", "b3": "Sw",
        }  # fmt: skip
        game = position(3, stacks)
        assert game.winning_move(Color.WHITE) == PlaceMove(sq("c3"))
        # Black filling c3 reaches only 2 to 3, so Black has no threat.
        assert not game.threatens_win(Color.BLACK)


# (size, layout, reserves, whether White threatens to win) for the symmetry check.
CASES = [
    (3, {"a2": "w w w"}, {}, True),
    (3, {"a1": "w", "b1": "Sb", "c1": "w", "b2": "b", "b3": "b"}, {}, False),
    (3, {"a1": "b", "b1": "b", "c1": "b w w", "a3": "w", "b3": "w", "c3": "b"}, {}, True),
    (3, {"a1": "w", "c1": "w", "b2": "b"}, {"white": (1, 0)}, True),
    (5, {"a1": "w", "b1": "w", "c1": "Sb", "d1": "w", "e1": "w", "c2": "Cw"}, {}, True),
]


@pytest.mark.parametrize(("size", "stacks", "reserves", "expected"), CASES)
def test_black_threats_mirror_white_threats(
    size: int,
    stacks: dict[str, str],
    reserves: dict[str, tuple[int, int]],
    expected: bool,
    position: Position,
    swap_colors: SwapColors,
) -> None:
    assert position(size, stacks, **reserves).threatens_win(Color.WHITE) is expected
    swapped = {"black" if name == "white" else "white": value for name, value in reserves.items()}
    mirror = position(size, swap_colors(stacks), **swapped)
    assert mirror.threatens_win(Color.BLACK) is expected


class TestQuerying:
    def test_the_game_is_left_unchanged(self, started: Started, snapshot: Snapshot) -> None:
        game = started(TAK_GAME, size=3)
        before = snapshot(game)
        assert game.threatens_win(Color.WHITE)
        assert not game.threatens_win(Color.BLACK)
        assert snapshot(game) == before
        assert game.to_move is Color.BLACK

    def test_no_threats_once_the_game_is_over(self, started: Started) -> None:
        game = started("a1 a2 b2 a3 c2", size=3)
        assert game.is_over
        assert game.winning_move(Color.WHITE) is None
        assert not game.threatens_win(Color.BLACK)

    def test_the_winning_move_really_wins(self, started: Started) -> None:
        game = started(TAK_GAME + " a3", size=3)  # Black ignores the threat
        move = game.winning_move(Color.WHITE)
        assert move is not None
        game.play(move)
        assert game.result is not None and game.result.winner is Color.WHITE


def random_positions(size: int, seed: int, max_plies: int) -> Iterator[Game]:
    rng = random.Random(seed * 7919 + size)
    game = Game(size)
    while not game.is_over and game.ply < max_plies:
        game.play(rng.choice(game.legal_moves()))
        yield game


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("seed", range(4))
def test_threats_agree_with_actually_playing_every_move(
    size: int, seed: int, wins_when_played: WinsWhenPlayed
) -> None:
    """The player to move threatens a win exactly when one of their moves wins."""
    for game in random_positions(size, seed, max_plies=60):
        if game.is_over or game.in_opening:
            continue
        mover = game.to_move
        expected = any(wins_when_played(game, mover, move) for move in game.legal_moves())
        assert game.threatens_win(mover) == expected
        winner = game.winning_move(mover)
        assert (winner is not None) == expected
        if winner is not None:
            assert wins_when_played(game, mover, winner)
