"""Tests for spotting a threat to win next turn ("Tak")."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from tak.engine import (
    Color,
    Direction,
    Game,
    Move,
    Piece,
    PieceType,
    PlaceMove,
    Reserve,
    SlideMove,
    Square,
)

Started = Callable[..., Game]
Snapshot = Callable[[Game], Any]

_LABELS = {
    "w": Piece(Color.WHITE, PieceType.FLAT),
    "b": Piece(Color.BLACK, PieceType.FLAT),
    "Sw": Piece(Color.WHITE, PieceType.WALL),
    "Sb": Piece(Color.BLACK, PieceType.WALL),
    "Cw": Piece(Color.WHITE, PieceType.CAP),
    "Cb": Piece(Color.BLACK, PieceType.CAP),
}


def position(size: int, stacks: dict[str, str], **reserves: tuple[int, int]) -> Game:
    """A game set up directly: ``{"c1": "b w w"}`` is a stack, bottom first.

    Reserves default to full; override with e.g. ``white=(1, 0)`` (stones, caps).
    The game is past the opening, with White to move.
    """
    game = Game(size)
    for name, labels in stacks.items():
        game.board.set_stack(Square.from_name(name), tuple(_LABELS[x] for x in labels.split()))
    for color in Color:
        if color.value in reserves:
            game.reserves[color] = Reserve(*reserves[color.value])
    return game


def wins_when_played(game: Game, color: Color, move: Move) -> bool:
    """Play ``move`` for ``color`` on a copy of the board and see if it wins."""
    assert color is game.to_move
    game.play(move)
    try:
        return game.result is not None and game.result.winner is color
    finally:
        game.undo()


class TestRoadThreats:
    def test_one_placement_from_a_road(self, started: Started) -> None:
        # White has b1, b2 and c3; c2 would join rank 1 to rank 3.
        game = started("a1 c3 b1 b3 b2", size=3)
        assert game.to_move is Color.BLACK
        assert game.threatens_win(Color.WHITE)
        assert game.winning_move(Color.WHITE) == PlaceMove(Square.from_name("c2"))

    def test_the_threat_is_judged_for_the_player_asked_about(self, started: Started) -> None:
        game = started("a1 c3 b1 b3 b2", size=3)
        assert not game.threatens_win(Color.BLACK)  # Black has only a1 and b3

    def test_no_threat_at_the_start(self, started: Started) -> None:
        for moves in ["", "a1", "a1 c3", "a1 c3 b1"]:
            game = started(moves, size=3)
            assert not game.threatens_win(Color.WHITE)
            assert not game.threatens_win(Color.BLACK)

    def test_two_placements_short_is_not_a_threat(self) -> None:
        game = position(5, {"a1": "w", "b1": "w", "c1": "w"})
        assert not game.threatens_win(Color.WHITE)

    def test_a_wall_in_the_way_blocks_the_threat(self) -> None:
        game = position(3, {"a1": "w", "b1": "Sb", "c1": "w", "b2": "b", "b3": "b"})
        assert not game.threatens_win(Color.WHITE)

    def test_a_slide_that_spreads_a_stack_into_a_road(self) -> None:
        # Only spreading a2 across b2 and c2 finishes a road in one move.
        game = position(3, {"a2": "w w w"})
        move = game.winning_move(Color.WHITE)
        assert move == SlideMove(Square.from_name("a2"), Direction.RIGHT, (1, 1))

    def test_a_capstone_flattening_a_wall_completes_the_road(self) -> None:
        game = position(5, {"a1": "w", "b1": "w", "c1": "Sb", "d1": "w", "e1": "w", "c2": "Cw"})
        move = game.winning_move(Color.WHITE)
        assert move == SlideMove(Square.from_name("c2"), Direction.DOWN, (1,))

    def test_completing_only_the_opponents_road_is_not_a_win(self) -> None:
        # White can uncover c1 to finish Black's road, but White has no road to make.
        game = position(3, {"a1": "b", "b1": "b", "c1": "b w", "a3": "Sb", "b3": "Sb"})
        assert not game.threatens_win(Color.WHITE)

    def test_completing_both_roads_is_a_win_for_the_mover(self) -> None:
        # 2c1+11 finishes White's top-row road but also uncovers Black's bottom row.
        # The mover wins a double road, and no other move wins, so this is the threat.
        game = position(3, {"a1": "b", "b1": "b", "c1": "b w w", "a3": "w", "b3": "w", "c3": "b"})
        move = game.winning_move(Color.WHITE)
        assert move == SlideMove(Square.from_name("c1"), Direction.UP, (1, 1))


class TestFlatThreats:
    # White leads on flats, 2 to 1, with nothing close to a road.
    STACKS = {"a1": "w", "c1": "w", "b2": "b"}

    def test_placing_the_last_piece_while_ahead(self) -> None:
        game = position(3, self.STACKS, white=(1, 0))
        assert game.threatens_win(Color.WHITE)

    def test_placing_the_last_piece_while_behind(self) -> None:
        # Black's last flat would only tie it 2 to 2: a draw is not a win.
        game = position(3, self.STACKS, black=(1, 0))
        assert not game.threatens_win(Color.BLACK)

    def test_placing_the_last_piece_to_go_ahead(self) -> None:
        # With c3 too, Black's last flat makes it 3 to 2 for Black.
        game = position(3, {**self.STACKS, "c3": "b"}, black=(1, 0))
        assert game.threatens_win(Color.BLACK)

    def test_filling_the_board_while_ahead(self) -> None:
        stacks = {
            "a1": "w", "b1": "Sb", "c1": "w",
            "a2": "Sb", "b2": "w", "c2": "Sb",
            "a3": "b", "b3": "Sw",
        }  # fmt: skip
        game = position(3, stacks)
        assert game.winning_move(Color.WHITE) == PlaceMove(Square.from_name("c3"))
        # Black filling c3 reaches only 2 to 3, so Black has no threat.
        assert not game.threatens_win(Color.BLACK)


class TestQuerying:
    def test_the_game_is_left_unchanged(self, started: Started, snapshot: Snapshot) -> None:
        game = started("a1 c3 b1 b3 b2", size=3)
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
        game = started("a1 c3 b1 b3 b2 a3", size=3)  # Black ignores the threat
        move = game.winning_move(Color.WHITE)
        assert move is not None
        game.play(move)
        assert game.result is not None and game.result.winner is Color.WHITE


class TestTinue:
    # 4x4: White has rows 2 and 3 up to the c-file, so d2 and d3 both win.
    DOUBLE = {"a2": "w", "b2": "w", "c2": "w", "a3": "w", "b3": "w", "c3": "w"}

    def test_a_single_threat_can_be_blocked(self, started: Started) -> None:
        game = started("a1 c3 b1 b3 b2", size=3)  # only c2 wins, and Black can take it
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    def test_two_threats_cannot_both_be_blocked(self) -> None:
        game = position(4, {**self.DOUBLE, "d1": "b"})
        assert game.is_tinue(Color.WHITE)

    def test_one_slide_can_block_two_threats(self) -> None:
        # With two stones on d1, Black can spread them over d2 and d3 (2d1+11).
        game = position(4, {**self.DOUBLE, "d1": "b b"})
        assert game.threatens_win(Color.WHITE)
        assert not game.is_tinue(Color.WHITE)

    def test_no_tinue_when_the_opponent_can_win_first(self) -> None:
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

    def test_the_double_threat_in_the_flat_race(self) -> None:
        assert position(4, self.FLAT_RACE).is_tinue(Color.WHITE)

    def test_a_reply_that_ends_in_a_draw_escapes(self) -> None:
        # Black's last piece ends the game level on flats, 6 to 6.
        game = position(4, self.FLAT_RACE, black=(1, 0))
        assert not game.is_tinue(Color.WHITE)

    def test_a_reply_that_loses_on_the_spot_does_not_escape(self) -> None:
        # Without c1, Black's last piece ends the game behind on flats, 5 to 6.
        stacks = {k: v for k, v in self.FLAT_RACE.items() if k != "c1"}
        game = position(4, stacks, black=(1, 0))
        assert game.is_tinue(Color.WHITE)

    def test_no_threat_means_no_tinue(self, started: Started) -> None:
        game = started("a1 c3 b1 b3", size=3)
        assert not game.threatens_win(Color.BLACK)
        assert not game.is_tinue(Color.BLACK)

    def test_no_tinue_once_the_game_is_over(self, started: Started) -> None:
        game = started("a1 a2 b2 a3 c2", size=3)
        assert not game.is_tinue(Color.WHITE)

    def test_the_game_is_left_unchanged(self, snapshot: Snapshot) -> None:
        game = position(4, {**self.DOUBLE, "d1": "b b"})
        before = snapshot(game)
        game.is_tinue(Color.WHITE)
        assert snapshot(game) == before


def random_positions(size: int, seed: int, max_plies: int) -> Iterator[Game]:
    rng = random.Random(seed * 7919 + size)
    game = Game(size)
    while not game.is_over and game.ply < max_plies:
        game.play(rng.choice(game.legal_moves()))
        yield game


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("seed", range(4))
def test_threats_agree_with_actually_playing_every_move(size: int, seed: int) -> None:
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


def tinue_by_brute_force(game: Game, threatener: Color) -> bool:
    """Tinue worked out with real moves only: the opponent is to move, and
    after every reply the threatener has lost nothing, i.e. has already won or
    has a move that wins."""
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
def test_tinue_agrees_with_playing_out_every_reply(size: int, seed: int) -> None:
    for game in random_positions(size, seed, max_plies=60):
        if game.is_over or game.in_opening:
            continue
        threatener = game.to_move.opponent  # the player who just moved
        if not game.threatens_win(threatener):
            assert not game.is_tinue(threatener)
            continue
        assert game.is_tinue(threatener) == tinue_by_brute_force(game, threatener)
