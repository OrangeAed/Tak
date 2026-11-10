"""Spotting "Tinue": a threat to win that the opponent cannot stop.

A move makes Tinue if it makes Tak and every reply the opponent could make
leaves the threatening player a winning move (or hands them the game on the
spot). A reply that wins for the opponent, or ends the game in a draw, stops it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from tak.engine import Color, Game, Move

Started = Callable[..., Game]
Snapshot = Callable[[Game], object]
Position = Callable[..., Game]
WinsWhenPlayed = Callable[[Game, Color, Move], bool]
RandomPositions = Callable[[int, int, int], Iterator[Game]]

# 4x4: White has rows 2 and 3 up to the c-file, so d2 and d3 both win.
DOUBLE = {"a2": "w", "b2": "w", "c2": "w", "a3": "w", "b3": "w", "c3": "w"}


class TestBlocking:
    def test_a_single_threat_can_be_blocked(self, started: Started) -> None:
        game = started("a1 c3 b1 b3 b2", size=3)  # only c2 wins, and Black can take it
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    @pytest.mark.parametrize("size", [3, 4, 5, 6, 7, 8])
    def test_a_single_threat_is_never_tinue(self, size: int, position: Position) -> None:
        files = "abcdefgh"[: size - 1]
        game = position(size, {f"{file}1": "w" for file in files})
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    def test_two_threats_cannot_both_be_blocked(self, position: Position) -> None:
        assert position(4, {**DOUBLE, "d1": "b"}).is_tinue(Color.WHITE)

    def test_black_can_make_tinue_too(self, position: Position) -> None:
        black_double = {square: "b" for square in DOUBLE}
        game = position(4, {**black_double, "d1": "w"})
        assert game.is_tinue(Color.BLACK)
        assert not game.is_tinue(Color.WHITE)

    def test_one_slide_can_block_two_threats(self, position: Position) -> None:
        # With two stones on d1, Black can spread them over d2 and d3 (2d1+11).
        game = position(4, {**DOUBLE, "d1": "b b"})
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    # 4x4: a2-b2-c2 reaches the d-file at d2, or through c3 at d3; both need c2.
    CORNER = {"a2": "w", "b2": "w", "c2": "w", "c3": "w", "a4": "b"}

    def test_two_threats_through_one_stone(self, position: Position) -> None:
        assert position(4, self.CORNER).is_tinue(Color.WHITE)

    def test_capturing_the_shared_stone_stops_both(self, position: Position) -> None:
        # Black's c1 can move onto c2, breaking both roads at once.
        game = position(4, {**self.CORNER, "c1": "b"})
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)


class TestCapstones:
    # 5x5: White needs e1. The capstone on e2 can take it even if a wall goes
    # there; the white wall on d2 stops e2 joining the road another way.
    STACKS = {"a1": "w", "b1": "w", "c1": "w", "d1": "w", "e2": "Cw", "d2": "Sw"}

    def test_a_wall_cannot_stop_a_capstone(self, position: Position) -> None:
        game = position(5, self.STACKS, black=(21, 0))  # Black has no capstone left
        assert game.is_tinue(Color.WHITE)

    def test_a_capstone_can_block_a_capstone(self, position: Position) -> None:
        game = position(5, self.STACKS)  # Black can put a capstone on e1
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    def test_without_the_capstone_a_wall_blocks(self, position: Position) -> None:
        stacks = {k: v for k, v in self.STACKS.items() if k != "e2"}
        assert not position(5, stacks, black=(21, 0)).is_tinue(Color.WHITE)


class TestGameEndingReplies:
    def test_no_tinue_when_the_opponent_can_win_first(self, position: Position) -> None:
        # 3x3: White threatens c1, c3 and a2, but Black wins at once with c2.
        stacks = {"a1": "w", "b1": "w", "a3": "w", "b3": "w", "a2": "b", "b2": "b"}
        game = position(3, stacks)
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    # Black has five flats to White's six; walls stop Black from threatening a road.
    FLAT_RACE = {
        **DOUBLE,
        "a1": "b", "b1": "b", "c1": "b", "d1": "Sw",
        "a4": "b", "b4": "b", "c4": "Sw",
    }  # fmt: skip

    def test_the_double_threat_in_the_flat_race(self, position: Position) -> None:
        assert position(4, self.FLAT_RACE).is_tinue(Color.WHITE)

    def test_a_reply_that_ends_in_a_draw_escapes(self, position: Position) -> None:
        # Black's last piece ends the game level on flats, 6 to 6.
        assert not position(4, self.FLAT_RACE, black=(1, 0)).is_tinue(Color.WHITE)

    def test_a_reply_that_loses_on_the_spot_does_not_escape(self, position: Position) -> None:
        # Without c1, Black's last piece ends the game behind on flats, 5 to 6.
        stacks = {k: v for k, v in self.FLAT_RACE.items() if k != "c1"}
        assert position(4, stacks, black=(1, 0)).is_tinue(Color.WHITE)


class TestQuerying:
    def test_no_threat_means_no_tinue(self, started: Started) -> None:
        game = started("a1 c3 b1 b3", size=3)
        assert not game.threatens_win(Color.BLACK)
        assert not game.is_tinue(Color.BLACK)

    def test_no_tinue_once_the_game_is_over(self, started: Started) -> None:
        assert not started("a1 a2 b2 a3 c2", size=3).is_tinue(Color.WHITE)

    def test_the_game_is_left_unchanged(self, position: Position, snapshot: Snapshot) -> None:
        for stacks in [{**DOUBLE, "d1": "b"}, {**DOUBLE, "d1": "b b"}]:
            game = position(4, stacks)
            before = snapshot(game)
            game.is_tinue(Color.WHITE)
            assert snapshot(game) == before


@pytest.fixture
def tinue_by_brute_force(wins_when_played: WinsWhenPlayed) -> Callable[[Game, Color], bool]:
    """Tinue worked out with real moves only, independent of the lookahead code.

    The opponent is to move. It's Tinue if after every reply the threatener has
    already won, or has a move that wins when really played.
    """

    def check(game: Game, threatener: Color) -> bool:
        assert game.to_move is threatener.opponent
        for reply in game.legal_moves():
            game.play(reply)
            try:
                if game.result is not None:
                    if game.result.winner is not threatener:
                        return False  # the opponent won or drew
                elif not any(
                    wins_when_played(game, threatener, move) for move in game.legal_moves()
                ):
                    return False
            finally:
                game.undo()
        return True

    return check


@pytest.mark.parametrize("size", [3, 4])
@pytest.mark.parametrize("seed", range(20))  # enough games to meet a dozen or so tinues
def test_tinue_agrees_with_playing_out_every_reply(
    size: int,
    seed: int,
    random_positions: RandomPositions,
    tinue_by_brute_force: Callable[[Game, Color], bool],
) -> None:
    for game in random_positions(size, seed, 60):
        if game.is_over or game.in_opening:
            continue
        threatener = game.to_move.opponent  # the player who just moved
        if not game.threatens_win(threatener):
            assert not game.is_tinue(threatener)
            continue
        assert game.is_tinue(threatener) == tinue_by_brute_force(game, threatener)
