"""Tests for spotting a threat to win that can't be stopped ("Tinue").

A threat is tinue if, whatever the opponent replies, the threatening player
can still win on their next move. A reply that wins or draws for the opponent
stops it; a reply that loses on the spot does not.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from tak.engine import Color, Game, Move, Piece, PieceType, PlaceMove, Reserve, Square, parse_move

Position = Callable[..., Game]
Started = Callable[..., Game]
Snapshot = Callable[[Game], Any]
WinsWhenPlayed = Callable[[Game, Color, Move], bool]
SwapColors = Callable[[dict[str, str]], dict[str, str]]

# 4x4: White has rows 2 and 3 up to the c-file, so d2 and d3 both win.
DOUBLE = {"a2": "w", "b2": "w", "c2": "w", "a3": "w", "b3": "w", "c3": "w"}

# The double threat again, with Black on five flats to White's six. Walls stop
# Black from threatening a road of their own.
FLAT_RACE = {
    **DOUBLE,
    "a1": "b", "b1": "b", "c1": "b", "d1": "Sw",
    "a4": "b", "b4": "b", "c4": "Sw",
}  # fmt: skip


class TestRoadThreats:
    def test_a_single_threat_can_be_blocked(self, started: Started) -> None:
        game = started("a1 c3 b1 b3 b2", size=3)  # only c2 wins, and Black can take it
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    def test_two_threats_cannot_both_be_blocked(self, position: Position) -> None:
        assert position(4, {**DOUBLE, "d1": "b"}).is_tinue(Color.WHITE)

    def test_one_slide_can_block_two_threats(self, position: Position) -> None:
        # With two stones on d1, Black can spread them over d2 and d3 (2d1+11).
        game = position(4, {**DOUBLE, "d1": "b b"})
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    def test_no_tinue_when_the_opponent_can_win_first(self, position: Position) -> None:
        # 3x3: White threatens c1, c3 and a2, but Black wins at once with c2.
        stacks = {"a1": "w", "b1": "w", "a3": "w", "b3": "w", "a2": "b", "b2": "b"}
        game = position(3, stacks)
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    def test_a_block_that_opens_a_different_win(self, started: Started) -> None:
        # White's only winning move is c3. Black can take c3, but that lets
        # White win with a2 instead, which didn't win before.
        game = started("c3 c2 b3 a1 c2< c3- Sc1 Sb1 a3", size=3)
        assert game.winning_move(Color.WHITE) == PlaceMove(Square.from_name("c3"))
        assert game.is_tinue(Color.WHITE)
        game.play(parse_move("c3"))
        assert game.winning_move(Color.WHITE) == PlaceMove(Square.from_name("a2"))

    def test_a_winning_slide_made_illegal_by_the_reply(self, started: Started) -> None:
        # Black threatens, among other things, b1< onto a1. White's wall on a1
        # stops that slide (and every other win), so this isn't tinue. Were the
        # slide made anyway, onto the wall, it would complete Black's road.
        game = started("a3 c1 b2 b1 b3 a2", size=3)
        assert game.threatens_win(Color.BLACK)
        assert not game.is_tinue(Color.BLACK)
        game.play(parse_move("Sa1"))
        assert parse_move("b1<") not in game.legal_moves()
        assert game.winning_move(Color.BLACK) is None


class TestFlatThreats:
    def test_the_double_threat_in_the_flat_race(self, position: Position) -> None:
        assert position(4, FLAT_RACE).is_tinue(Color.WHITE)

    def test_a_reply_that_ends_in_a_draw_escapes(self, position: Position) -> None:
        # Black's last piece ends the game level on flats, 6 to 6.
        assert not position(4, FLAT_RACE, black=(1, 0)).is_tinue(Color.WHITE)

    def test_a_reply_that_loses_on_the_spot_does_not_escape(self, position: Position) -> None:
        # Without c1, Black's last piece ends the game behind on flats, 5 to 6.
        stacks = {name: labels for name, labels in FLAT_RACE.items() if name != "c1"}
        assert position(4, stacks, black=(1, 0)).is_tinue(Color.WHITE)

    # No roads anywhere near: White's last stone would win on flats, 4 to 1.
    LAST_STONE = {"b2": "w", "c2": "w", "d4": "w"}

    def test_the_last_stone_wins_whatever_black_does(self, position: Position) -> None:
        # Each Black reply narrows the gap by at most one.
        game = position(4, {**self.LAST_STONE, "a2": "b"}, white=(1, 0))
        assert game.is_tinue(Color.WHITE)

    def test_a_stack_covering_two_flats_levels_the_count(self, position: Position) -> None:
        # 2a2>11 covers b2 and c2, so White's last stone only draws, 2 to 2.
        game = position(4, {**self.LAST_STONE, "a2": "b b"}, white=(1, 0))
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)


class TestLargeBoards:
    """An 8x8 double threat where Black has thousands of replies to try."""

    def build(self, walled: bool) -> Game:
        game = Game(8)
        white, black = Piece(Color.WHITE, PieceType.FLAT), Piece(Color.BLACK, PieceType.FLAT)
        for file in range(7):  # ranks 2 and 4 reach the g-file: h2 and h4 win
            game.board.set_stack(Square(file, 1), (white,))
            game.board.set_stack(Square(file, 3), (white,))
        if walled:  # a row of walls on rank 5 keeps Black's stacks away
            for file in range(8):
                game.board.set_stack(Square(file, 4), (Piece(Color.WHITE, PieceType.WALL),))
        for file, rank in [(1, 6), (3, 6), (5, 6), (2, 7), (6, 7), (0, 5), (4, 5), (7, 6)]:
            game.board.set_stack(Square(file, rank), (white, black) * 4 + (black,))
        game.reserves = {Color.WHITE: Reserve(2, 2), Color.BLACK: Reserve(10, 2)}
        return game

    def test_walled_off_stacks_cannot_help(self) -> None:
        assert self.build(walled=True).is_tinue(Color.WHITE)

    def test_a_tall_stack_can_spread_over_both_roads(self) -> None:
        game = self.build(walled=False)
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)


# (size, layout, reserves, whether White's threat is tinue) for the symmetry check.
CASES = [
    (4, {**DOUBLE, "d1": "b"}, {}, True),
    (4, {**DOUBLE, "d1": "b b"}, {}, False),
    (4, FLAT_RACE, {"black": (1, 0)}, False),
    (4, {"b2": "w", "c2": "w", "d4": "w", "a2": "b"}, {"white": (1, 0)}, True),
]


@pytest.mark.parametrize(("size", "stacks", "reserves", "expected"), CASES)
def test_black_tinue_mirrors_white_tinue(
    size: int,
    stacks: dict[str, str],
    reserves: dict[str, tuple[int, int]],
    expected: bool,
    position: Position,
    swap_colors: SwapColors,
) -> None:
    assert position(size, stacks, **reserves).is_tinue(Color.WHITE) is expected
    swapped = {"black" if name == "white" else "white": value for name, value in reserves.items()}
    mirror = position(size, swap_colors(stacks), **swapped)
    assert mirror.is_tinue(Color.BLACK) is expected


class TestQuerying:
    def test_no_threat_means_no_tinue(self, started: Started) -> None:
        game = started("a1 c3 b1 b3", size=3)
        assert not game.threatens_win(Color.BLACK)
        assert not game.is_tinue(Color.BLACK)

    def test_no_tinue_once_the_game_is_over(self, started: Started) -> None:
        assert not started("a1 a2 b2 a3 c2", size=3).is_tinue(Color.WHITE)

    def test_the_game_is_left_unchanged(self, position: Position, snapshot: Snapshot) -> None:
        game = position(4, {**DOUBLE, "d1": "b b"})
        before = snapshot(game)
        game.is_tinue(Color.WHITE)
        assert snapshot(game) == before


def random_positions(size: int, seed: int, max_plies: int) -> Iterator[Game]:
    rng = random.Random(seed * 7919 + size)
    game = Game(size)
    while not game.is_over and game.ply < max_plies:
        game.play(rng.choice(game.legal_moves()))
        yield game


def tinue_by_brute_force(game: Game, threatener: Color, wins_when_played: WinsWhenPlayed) -> bool:
    """Tinue worked out with real moves only. The opponent is to move; after
    every reply the threatener must already have won or have a winning move."""
    assert game.to_move is threatener.opponent
    for reply in game.legal_moves():
        game.play(reply)
        try:
            if game.result is not None:
                if game.result.winner is not threatener:
                    return False  # the opponent won or drew
            elif not any(wins_when_played(game, threatener, move) for move in game.legal_moves()):
                return False
        finally:
            game.undo()
    return True


@pytest.mark.parametrize("size", [3, 4])
@pytest.mark.parametrize("seed", range(20))  # enough games to meet a dozen or so tinues
def test_tinue_agrees_with_playing_out_every_reply(
    size: int, seed: int, wins_when_played: WinsWhenPlayed
) -> None:
    for game in random_positions(size, seed, max_plies=60):
        if game.is_over or game.in_opening:
            continue
        threatener = game.to_move.opponent  # the player who just moved
        if not game.threatens_win(threatener):
            assert not game.is_tinue(threatener)
            continue
        expected = tinue_by_brute_force(game, threatener, wins_when_played)
        assert game.is_tinue(threatener) == expected
