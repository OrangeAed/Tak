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

from tak.engine import (
    Color,
    Game,
    Move,
    Piece,
    PieceType,
    PlaceMove,
    Reserve,
    Square,
    format_move,
    parse_move,
)

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


class TestBlocksThatOpenNewThreats:
    """A reply can stop every threat the player has now, yet open a new one.

    Whether that reply saves the game depends on the new threat, not the old
    ones. So it's tinue when every reply that stops the old threats opens a
    new one, and merely tak if some other reply stops everything for good.

    Each case is checked with real moves: the threatener's winning moves are
    found by playing every move from a copy of the position where it's their
    turn, and the replies are really played.
    """

    def winning_moves(self, game: Game, color: Color, wins: WinsWhenPlayed) -> set[str]:
        """Every move that wins for ``color``, who must be to move."""
        return {format_move(move) for move in game.legal_moves() if wins(game, color, move)}

    def check(
        self,
        position: Position,
        wins_when_played: WinsWhenPlayed,
        setup: tuple[int, dict[str, str]],
        threatener: Color,
        threats: set[str],
        backfire: str,
        new_threats: set[str] | None,
        escape: str | None,
        **reserves: tuple[int, int],
    ) -> None:
        """Check one case.

        ``threats`` are the threatener's winning moves now. ``backfire`` is a
        reply that stops them all, after which the threatener can win with
        ``new_threats`` (or with at least one move, if that's ``None``), none
        of which won before. ``escape`` is a reply that stops everything, or
        ``None`` if there isn't one, in which case the threat is tinue.
        ``reserves`` are passed on to ``position``.
        """
        size, stacks = setup
        opponent = threatener.opponent
        before = position(size, stacks, to_move=threatener, **reserves)
        assert self.winning_moves(before, threatener, wins_when_played) == threats

        game = position(size, stacks, to_move=opponent, **reserves)
        assert game.threatens_win(threatener)

        game.play(parse_move(backfire))
        assert game.result is None
        after = self.winning_moves(game, threatener, wins_when_played)
        assert after, "the backfiring reply should leave a way to win"
        assert not after & threats, "the backfiring reply should stop every old threat"
        if new_threats is not None:
            assert after == new_threats
        game.undo()

        if escape is not None:
            game.play(parse_move(escape))
            assert game.result is None
            assert not self.winning_moves(game, threatener, wins_when_played)
            game.undo()

        expected = escape is None
        assert game.is_tinue(threatener) is expected
        assert tinue_by_brute_force(game, threatener, wins_when_played) is expected

    # --- Uncovering: moving off a stack reveals the threatener's piece ----------

    def test_a_block_that_uncovers_a_flat_is_a_trap(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # 3x3. White threatens a2. Black's b2< blocks a2, but uncovers White's
        # flat on b2, which can then capture a2. Taking the whole stack (2b2<)
        # blocks without uncovering anything.
        #   3 | w  | .  | .  |
        #   2 | .  | wb | .  |
        #   1 | w  | .  | .  |
        self.check(
            position,
            wins_when_played,
            (3, {"a1": "w", "a3": "w", "b2": "w b"}),
            Color.WHITE,
            threats={"a2"},
            backfire="b2<",
            new_threats={"b2<"},
            escape="2b2<",
        )

    # 4x4. White's rows 2 and 3 each need the d-file square. Black's stack on d1
    # can spread over both d2 and d3, but each spread that does leaves a white
    # flat on top of d1, and then c1 finishes White's bottom row. Walls on
    # rank 4 stop White from threatening the columns.
    #   4 | Sb | Sb | Sb | .  |
    #   3 | w  | w  | w  | .  |
    #   2 | w  | w  | w  | .  |
    #   1 | w  | w  | .  | d1 |
    UNDER_A_STACK = {
        "a1": "w", "b1": "w",
        "a2": "w", "b2": "w", "c2": "w",
        "a3": "w", "b3": "w", "c3": "w",
        "a4": "Sb", "b4": "Sb", "c4": "Sb",
    }  # fmt: skip

    def test_a_stack_that_can_be_carried_off_whole_is_a_trap(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # d1 holds w b b. Spreading only the two black stones uncovers the
        # white one, but all three can be carried off (3d1+21), burying it.
        self.check(
            position,
            wins_when_played,
            (4, {**self.UNDER_A_STACK, "d1": "w b b"}),
            Color.WHITE,
            threats={"d2", "d3"},
            backfire="2d1+11",
            new_threats=None,
            escape="3d1+21",
        )

    def test_a_stack_too_tall_to_carry_off_is_tinue(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # d1 holds w w w w b b: six stones, but at most four can be carried, so
        # every spread over d2 and d3 leaves a white flat on d1. (Spreads that
        # drop a white stone on top of d2 complete White's road at once.)
        for backfire in ["2d1+11", "3d1+21", "4d1+31"]:
            self.check(
                position,
                wins_when_played,
                (4, {**self.UNDER_A_STACK, "d1": "w w w w b b"}),
                Color.WHITE,
                threats={"d2", "d3"},
                backfire=backfire,
                new_threats=None,
                escape=None,
            )

    # 3x3. White threatens a2. Black's stack on b2 holds five stones, more than
    # the three that can be carried, so every slide onto a2 leaves a white flat
    # on b2 that can capture a2 back. White walls on b1 and b3 mean the only new
    # wins are such slides, not placements.
    #   3 |   w   |  Sw   |   .   |
    #   2 |   .   | wwwwb |   w   |
    #   1 |   w   |  Sw   |   .   |
    TALL_STACK = {"a1": "w", "a3": "w", "c2": "w", "b1": "Sw", "b3": "Sw", "b2": "w w w w b"}

    def test_when_the_new_threats_are_all_captures_it_is_still_tinue(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # With one stone left and behind on flats (1 to 3), every placement by
        # Black ends the game in White's favour, so blocking means sliding.
        for backfire, new_threats in [
            ("b2<", {"b2<", "2b2<", "3b2<", "a1+", "a3-"}),
            ("2b2<", {"b2<", "2b2<", "3b2<", "a1+", "a3-"}),
            ("3b2<", {"b2<", "2b2<", "a1+", "a3-"}),
        ]:
            self.check(
                position,
                wins_when_played,
                (3, self.TALL_STACK),
                Color.WHITE,
                threats={"a2"},
                backfire=backfire,
                new_threats=new_threats,
                escape=None,
                black=(1, 0),
            )

    def test_the_same_block_with_stones_to_spare_is_only_tak(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # With stones in reserve, Black can simply place a wall on a2 instead.
        self.check(
            position,
            wins_when_played,
            (3, self.TALL_STACK),
            Color.WHITE,
            threats={"a2"},
            backfire="b2<",
            new_threats={"b2<", "2b2<", "3b2<", "a1+", "a3-"},
            escape="Sa2",
        )

    # --- Vacating: the blocking piece was holding back another road ------------

    def test_a_block_that_vacates_a_square_is_a_trap(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # 3x3. White threatens b3 (a3-b3-b2-c2) and a3- (capturing a2 for row 2).
        # Black's a2+ captures a3, stopping both, but leaves a2 empty for
        # White's row 2. Capturing b2 instead (b1+) stops everything.
        #   3 | w  | .  | .  |
        #   2 | b  | w  | w  |
        #   1 | .  | b  | .  |
        self.check(
            position,
            wins_when_played,
            (3, {"a3": "w", "a2": "b", "b2": "w", "c2": "w", "b1": "b"}),
            Color.WHITE,
            threats={"b3", "a3-"},
            backfire="a2+",
            new_threats={"a2"},
            escape="b1+",
        )

    def test_the_only_block_vacates_a_square_so_black_has_tinue(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # 3x3, Black threatening. Black threatens row 2 (b2, b3-, a2>) and row 3
        # (c2+ captures c3). White's only way to stop both is c3-, capturing
        # c2, but that leaves c3 empty, and Black's c3 completes row 3.
        #   3 | b  | b  | w  |
        #   2 | b  | .  | b  |
        #   1 | w  | Sw | .  |
        self.check(
            position,
            wins_when_played,
            (3, {"a3": "b", "b3": "b", "c3": "w", "a2": "b", "c2": "b", "a1": "w", "b1": "Sw"}),
            Color.BLACK,
            threats={"b2", "b3-", "a2>", "c2+"},
            backfire="c3-",
            new_threats={"c3"},
            escape=None,
        )

    # --- A different kind of win: blocking the road lets White fill the board --

    # 3x3. White threatens a1 for a road. Blocking it leaves c1 as the only
    # empty square, and filling the board there wins White the flat count.
    # (b2 and c3, marked ?, differ between the two tests.)
    #   3 | w  | Sw | ?  |
    #   2 | w  | ?  | b  |
    #   1 | .  | Sw | .  |
    FLAT_FINISH = {"a3": "w", "b3": "Sw", "a2": "w", "c2": "b", "b1": "Sw"}

    def test_blocking_the_road_leaves_a_flat_win_so_it_is_tinue(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # With walls on b2 and c3, Black can't reach White's road. Once a1 is
        # blocked, White filling c1 wins on flats: 3 to 2 after a flat on a1,
        # 3 to 1 after a wall (so White may even fill it with a wall then).
        for backfire, new_threats in [("a1", {"c1"}), ("Sa1", {"c1", "Sc1"})]:
            self.check(
                position,
                wins_when_played,
                (3, {**self.FLAT_FINISH, "b2": "Sw", "c3": "Sb"}),
                Color.WHITE,
                threats={"a1"},
                backfire=backfire,
                new_threats=new_threats,
                escape=None,
            )

    def test_blocking_the_road_leaves_a_flat_win_but_a_capture_escapes(
        self, position: Position, wins_when_played: WinsWhenPlayed
    ) -> None:
        # With a black flat on b2 (and a white one on c3, keeping White ahead),
        # blocking a1 still loses on flats, but b2< captures a2 instead.
        self.check(
            position,
            wins_when_played,
            (3, {**self.FLAT_FINISH, "b2": "b", "c3": "w"}),
            Color.WHITE,
            threats={"a1"},
            backfire="a1",
            new_threats={"c1"},
            escape="b2<",
        )

    # --- From a real game --------------------------------------------------------

    def test_every_block_backfires_in_a_real_game(
        self, started: Started, wins_when_played: WinsWhenPlayed
    ) -> None:
        # White threatens a2 (and more). Black filling a2 fills the board and
        # loses on flats; b1+ and c2< each block everything but vacate a square
        # that White then takes for a road.
        #   3 | Sb | w  | w  |
        #   2 | .  | w  | b  |
        #   1 | w  | b  | w  |
        game = started("c2 c3 c1 Sa2 b2 a2+ a1 b1 b3", size=3)
        assert game.is_tinue(Color.WHITE)
        assert tinue_by_brute_force(game, Color.WHITE, wins_when_played)
        for backfire, new_threat in [("b1+", "b1"), ("c2<", "c2")]:
            game.play(parse_move(backfire))
            assert self.winning_moves(game, Color.WHITE, wins_when_played) == {new_threat}
            game.undo()


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
