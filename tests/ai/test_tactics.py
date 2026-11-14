"""The fast look-ahead the computer players rely on, checked against the engine.

The engine answers the same questions by trying every move, so it serves as
the brute-force reference: on hand-built positions, on random positions, and
on positions from games between the computer players (which are full of
threats and Tinue).
"""

from __future__ import annotations

import pytest

from tak.ai.tactics import (
    find_win,
    groups,
    has_road,
    is_tinue,
    loses_or_allows_win,
    slides_along,
    spans,
)
from tak.engine import Color, Direction, Game, Move, PlaceMove, SlideMove, Square, parse_move

from .conftest import Position, bot_game_positions, random_positions

SIZES = [3, 4, 5, 6, 7, 8]


def snapshot(game: Game) -> tuple[object, ...]:
    stacks = tuple(game.board.stack(square) for square in game.board.squares())
    return stacks, dict(game.reserves), game.result, game.ply


def wins_for_mover(game: Game, move: Move) -> bool:
    """Whether ``move`` wins for the player to move, by really playing it."""
    mover = game.to_move
    game.play(move)
    try:
        return game.result is not None and game.result.winner is mover
    finally:
        game.undo()


def allows_loss_brute_force(game: Game, move: Move) -> bool:
    """Whether ``move`` loses on the spot or leaves the opponent a win, using the engine."""
    mover = game.to_move
    after = game.copy()
    after.play(move)
    if after.result is not None:
        return after.result.winner is mover.opponent
    return after.winning_move(mover.opponent) is not None


class TestRoadGroups:
    def test_groups_note_the_edges_they_touch(self) -> None:
        # 3x3: a1-a2 on the left edge, and c3 alone in the top right corner.
        cells = [True, False, False, True, False, False, False, False, True]
        label, flags = groups(cells, 3)
        assert label[0] == label[3] != label[8]
        assert label[1] == -1
        assert len(flags) == 2
        assert not any(spans(f) for f in flags)

    def test_a_full_column_is_a_road(self) -> None:
        cells = [False, True, False] * 3  # the b-file
        assert has_road(cells, 3)
        assert not has_road([True, False, False, False, True, False, False, False, True], 3)


class TestFindWin:
    def test_finds_a_road_completing_placement(self, position: Position) -> None:
        game = position(4, {"a2": "w", "b2": "w", "c2": "w", "a1": "b"})
        assert find_win(game, Color.WHITE) == PlaceMove(Square.from_name("d2"))
        assert find_win(game, Color.BLACK) is None

    def test_finds_a_road_completing_slide(self, position: Position) -> None:
        # 3x3: no placement makes a road, but b3- (covering b2) makes the b-file one.
        game = position(3, {"b1": "w", "b2": "b", "b3": "w w", "c1": "b"})
        move = find_win(game, Color.WHITE)
        assert isinstance(move, SlideMove) and move.start == Square.from_name("b3")
        assert wins_for_mover(game, move)

    def test_a_capstone_flattening_a_wall_completes_a_road(self, position: Position) -> None:
        game = position(3, {"a2": "w", "b2": "Sb", "c2": "w", "b1": "Cw", "a1": "b"})
        assert find_win(game, Color.WHITE) == parse_move("b1+")

    def test_walls_and_the_opponents_capstone_block(self, position: Position) -> None:
        game = position(3, {"a2": "w", "b2": "Cb", "c2": "w", "b1": "w w"})
        assert game.winning_move(Color.WHITE) is None
        assert find_win(game, Color.WHITE) is None

    def test_a_flat_win_by_placing_the_last_piece(self, position: Position) -> None:
        game = position(3, {"a1": "w", "c3": "w", "b3": "b"}, white=(1, 0))
        move = find_win(game, Color.WHITE)
        assert move is not None and wins_for_mover(game, move)

    def test_first_suggestions_are_checked_not_trusted(self, position: Position) -> None:
        game = position(4, {"a2": "w", "b2": "w", "c2": "w"})
        wrong = PlaceMove(Square.from_name("d4"))
        assert find_win(game, Color.WHITE, first=[wrong]) == PlaceMove(Square.from_name("d2"))

    def test_nothing_to_win_once_the_game_is_over(self) -> None:
        game = Game(3)
        for token in ["a1", "a2", "b2", "a3", "c2"]:
            game.play(parse_move(token))
        assert game.is_over
        assert find_win(game, Color.WHITE) is None

    @pytest.mark.parametrize("size", SIZES)
    def test_agrees_with_the_engine_on_random_positions(self, size: int) -> None:
        for game in random_positions(size, 25, seed=size):
            before = snapshot(game)
            for color in Color:
                fast = find_win(game, color)
                assert (fast is None) == (game.winning_move(color) is None)
            mover_win = find_win(game, game.to_move)
            if mover_win is not None:
                assert wins_for_mover(game, mover_win)
            assert snapshot(game) == before

    @pytest.mark.parametrize("size", [3, 4, 5])
    def test_agrees_with_the_engine_in_games_between_computer_players(self, size: int) -> None:
        for game in bot_game_positions(size, seed=size, white="strategist", black="greedy"):
            for color in Color:
                fast = find_win(game, color)
                assert (fast is None) == (game.winning_move(color) is None)


class TestSlidesAlong:
    @pytest.mark.parametrize("size", SIZES)
    def test_every_legal_slide_and_nothing_else(self, size: int) -> None:
        for game in random_positions(size, 10, seed=100 + size, slide_bias=0.8):
            mover = game.to_move
            expected = [m for m in game.legal_moves() if isinstance(m, SlideMove)]
            found = [
                move
                for square in game.board.squares()
                if (top := game.board.top(square)) is not None and top.color is mover
                for direction in Direction
                for move in slides_along(game.board, square, direction)
            ]
            assert sorted(map(repr, found)) == sorted(map(repr, expected))


class TestSafety:
    def test_a_move_that_ignores_a_threat_is_unsafe(self, position: Position) -> None:
        # Black threatens d2; only taking d2 (or a wall there) stops it.
        game = position(4, {"a2": "b", "b2": "b", "c2": "b", "a4": "w"})
        assert loses_or_allows_win(game, parse_move("a3"))
        assert not loses_or_allows_win(game, parse_move("d2"))
        assert not loses_or_allows_win(game, parse_move("Sd2"))

    def test_completing_only_the_opponents_road_loses(self, position: Position) -> None:
        # Lifting the white stone off b2 uncovers Black's flat and completes a2-b2-c2.
        game = position(3, {"a2": "b", "b2": "b w", "c2": "b", "a3": "w"})
        assert loses_or_allows_win(game, parse_move("b2+"))

    def test_threats_found_are_remembered(self, position: Position) -> None:
        game = position(4, {"a2": "b", "b2": "b", "c2": "b", "a4": "w"})
        threats: list[Move] = []
        loses_or_allows_win(game, parse_move("a3"), threats)
        assert threats == [PlaceMove(Square.from_name("d2"))]

    @pytest.mark.parametrize("size", [3, 4, 5])
    def test_agrees_with_the_engine(self, size: int) -> None:
        positions = random_positions(size, 8, seed=200 + size)
        positions += bot_game_positions(size, seed=size, white="greedy", black="strategist")[::3]
        for game in positions:
            before = snapshot(game)
            threats: list[Move] = []
            for move in game.legal_moves()[:60]:
                expected = allows_loss_brute_force(game, move)
                assert loses_or_allows_win(game, move, threats) == expected, move
            assert snapshot(game) == before


class TestTinue:
    DOUBLE = {"a2": "w", "b2": "w", "c2": "w", "a3": "w", "b3": "w", "c3": "w"}

    def test_two_threats_cannot_both_be_blocked(self, position: Position) -> None:
        game = position(4, {**self.DOUBLE, "d1": "b"}, to_move=Color.BLACK)
        assert is_tinue(game, Color.WHITE)

    def test_one_slide_can_block_two_threats(self, position: Position) -> None:
        game = position(4, {**self.DOUBLE, "d1": "b b"}, to_move=Color.BLACK)
        assert not is_tinue(game, Color.WHITE)

    def test_a_single_threat_is_not_tinue(self, position: Position) -> None:
        game = position(4, {"a2": "w", "b2": "w", "c2": "w", "a4": "b"}, to_move=Color.BLACK)
        assert not is_tinue(game, Color.WHITE)

    def test_hints_are_checked_not_trusted(self, position: Position) -> None:
        game = position(4, {"a2": "w", "b2": "w", "a4": "b"}, to_move=Color.BLACK)
        assert not is_tinue(game, Color.WHITE, hints=[PlaceMove(Square.from_name("d2"))])

    @pytest.mark.parametrize("size", [3, 4, 5])
    def test_agrees_with_the_engine_in_games_between_computer_players(self, size: int) -> None:
        positions = bot_game_positions(size, seed=7 * size, white="strategist", black="greedy")
        positions += bot_game_positions(size, seed=7 * size, white="greedy", black="strategist")
        tinues = 0
        for game in positions:
            just_moved = game.to_move.opponent
            before = snapshot(game)
            expected = game.is_tinue(just_moved)
            assert is_tinue(game, just_moved) == expected
            assert snapshot(game) == before
            tinues += expected
        assert tinues > 0  # the games really did test some Tinue positions
