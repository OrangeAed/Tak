"""Tests for PTN parsing and writing."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tak.engine import (
    Direction,
    Game,
    GameResult,
    IllegalMoveError,
    NotationError,
    PieceType,
    PlaceMove,
    Reason,
    SlideMove,
    Square,
    format_move,
    game_from_ptn,
    game_to_ptn,
    parse_move,
    parse_ptn,
)
from tak.engine.pieces import Color


def sq(name: str) -> Square:
    return Square.from_name(name)


class TestParseMove:
    @pytest.mark.parametrize(
        ("text", "piece"),
        [
            ("a1", PieceType.FLAT),
            ("Fa1", PieceType.FLAT),
            ("Sa1", PieceType.WALL),
            ("Ca1", PieceType.CAP),
        ],
    )
    def test_placements(self, text: str, piece: PieceType) -> None:
        assert parse_move(text) == PlaceMove(sq("a1"), piece)

    def test_placement_square(self) -> None:
        assert parse_move("Cd4") == PlaceMove(sq("d4"), PieceType.CAP)

    @pytest.mark.parametrize(
        ("text", "start", "direction", "drops"),
        [
            ("a1>", "a1", Direction.RIGHT, (1,)),
            ("4c3>", "c3", Direction.RIGHT, (4,)),
            ("3b2+111", "b2", Direction.UP, (1, 1, 1)),
            ("2d4-2", "d4", Direction.DOWN, (2,)),
            ("5e4<23", "e4", Direction.LEFT, (2, 3)),
            ("a1<", "a1", Direction.LEFT, (1,)),
        ],
    )
    def test_slides(
        self, text: str, start: str, direction: Direction, drops: tuple[int, ...]
    ) -> None:
        assert parse_move(text) == SlideMove(sq(start), direction, drops)

    def test_single_stone_may_state_its_drop(self) -> None:
        assert parse_move("a1>1") == parse_move("a1>")

    def test_arrow_directions(self) -> None:
        assert parse_move("c3→") == parse_move("c3>")
        assert parse_move("c3←") == parse_move("c3<")
        assert parse_move("c3↑") == parse_move("c3+")
        assert parse_move("3c3↓111") == parse_move("3c3-111")

    @pytest.mark.parametrize("mark", ["*", "'", "''", '"', "!", "?", "!!", "?!", "*'"])
    def test_annotation_marks_are_ignored(self, mark: str) -> None:
        assert parse_move("c3>" + mark) == parse_move("c3>")
        assert parse_move("Sb2" + mark) == parse_move("Sb2")

    def test_optional_top_piece_letter_is_accepted(self) -> None:
        assert parse_move("2c3>2S") == parse_move("2c3>2")
        assert parse_move("c3>C") == parse_move("c3>")

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert parse_move("  a1 \n") == parse_move("a1")

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "z9",
            "a",
            "a9",
            "A1",
            "a1x",
            "Ca1>",  # a placement letter can't prefix a slide
            "0a1>",  # can't pick up zero
            "9a1>",  # no such count
            "a1>0",  # a drop of zero
            "3a1",  # no direction
            "a1>12",  # drops add up to 3 but only 1 stone is picked up
            "3c2>13",  # drops add up to 4 but 3 stones are picked up
            "3c2>1",  # drops add up to 1 but 3 stones are picked up
            "move a1",
        ],
    )
    def test_invalid_moves_are_rejected(self, bad: str) -> None:
        with pytest.raises(NotationError):
            parse_move(bad)

    def test_drop_mismatch_message_is_helpful(self) -> None:
        with pytest.raises(NotationError, match="add up to 4.*3 stone"):
            parse_move("3c2>13")


class TestFormatMove:
    @pytest.mark.parametrize(
        "text",
        ["a1", "Sb2", "Cc3", "d4-", "3c2>", "4a1+211", "5e4<23", "2d4-", "a1<", "3b2+111"],
    )
    def test_canonical_text_round_trips(self, text: str) -> None:
        assert format_move(parse_move(text)) == text

    @pytest.mark.parametrize(
        ("written", "canonical"),
        [("Fa1", "a1"), ("a1>1", "a1>"), ("2d4-2", "2d4-"), ("c3→", "c3>"), ("2c3>2S!", "2c3>")],
    )
    def test_writing_normalizes_to_the_shortest_form(self, written: str, canonical: str) -> None:
        assert format_move(parse_move(written)) == canonical

    def test_flatten_mark(self) -> None:
        move = parse_move("c3>")
        assert format_move(move, flattened=True) == "c3>*"
        assert format_move(move) == "c3>"


class TestParsePtn:
    def test_headers_moves_comments_and_result(self) -> None:
        text = """
        [Event "Test"]
        [Size "3"]
        [Player1 "Alice"]

        1. a1 a3 {Opening placements}
        2. b2 b1 3. c2 R-0
        """
        parsed = parse_ptn(text)
        assert parsed.headers == {"Event": "Test", "Size": "3", "Player1": "Alice"}
        assert [format_move(m) for m in parsed.moves] == ["a1", "a3", "b2", "b1", "c2"]
        assert parsed.result == "R-0"

    def test_move_numbers_may_touch_the_move(self) -> None:
        parsed = parse_ptn("1.a1 a3 2.b2")
        assert [format_move(m) for m in parsed.moves] == ["a1", "a3", "b2"]

    def test_no_result(self) -> None:
        assert parse_ptn("1. a1 a3").result is None

    @pytest.mark.parametrize("token", ["R-0", "0-R", "F-0", "0-F", "1-0", "0-1", "1/2-1/2"])
    def test_every_result_token(self, token: str) -> None:
        assert parse_ptn(f"1. a1 a3 {token}").result == token

    def test_unreadable_move_raises(self) -> None:
        with pytest.raises(NotationError):
            parse_ptn("1. a1 nonsense")


ROAD_GAME = "1. a1 a2 2. b2 a3 3. c2"


class TestGames:
    def test_game_from_ptn_replays_every_move(self) -> None:
        game = game_from_ptn('[Size "3"]\n' + ROAD_GAME)
        assert game.size == 3
        assert game.ply == 5
        assert game.result == GameResult(Color.WHITE, Reason.ROAD)

    def test_game_to_ptn_writes_headers_moves_and_result(self) -> None:
        game = game_from_ptn('[Size "3"]\n' + ROAD_GAME)
        text = game_to_ptn(game, {"Player1": "Alice"})
        assert text == (
            '[Size "3"]\n[Player1 "Alice"]\n[Result "R-0"]\n\n1. a1 a2\n2. b2 a3\n3. c2 R-0\n'
        )

    def test_unfinished_game_has_no_result(self) -> None:
        game = game_from_ptn('[Size "5"]\n1. a1 e5 2. c3')
        text = game_to_ptn(game)
        assert "Result" not in text
        assert text.splitlines()[-1] == "2. c3"

    def test_ptn_survives_a_round_trip(self) -> None:
        original = game_from_ptn('[Size "3"]\n' + ROAD_GAME)
        again = game_from_ptn(game_to_ptn(original))
        assert [p.move for p in again.history] == [p.move for p in original.history]
        assert again.result == original.result

    def test_flattening_is_marked_with_an_asterisk(self, started: Callable[..., Game]) -> None:
        game = started("a1 e5 Cc3 d3 b5 Sd4 b4 d4- c3>")
        assert game.history[-1].flattened
        assert game_to_ptn(game).splitlines()[-1] == "5. c3>*"
        assert [p.flattened for p in game.history].count(True) == 1

    def test_size_header_is_required(self) -> None:
        with pytest.raises(NotationError, match="Size"):
            game_from_ptn("1. a1 a3")

    @pytest.mark.parametrize("size", ["x", "2", "9"])
    def test_bad_sizes_are_rejected(self, size: str) -> None:
        with pytest.raises(NotationError):
            game_from_ptn(f'[Size "{size}"]\n1. a1 a3')

    def test_tps_start_positions_are_not_supported_yet(self) -> None:
        with pytest.raises(NotationError, match="TPS"):
            game_from_ptn('[Size "5"]\n[TPS "x5/x5/x5/x5/x5 1 1"]\n1. a1')

    def test_illegal_moves_report_which_move_failed(self) -> None:
        with pytest.raises(IllegalMoveError, match=r"move 2 \(a1\).*occupied"):
            game_from_ptn('[Size "3"]\n1. a1 a1')
