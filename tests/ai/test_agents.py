"""The computer players: they play legal moves, and each plays the way it promises."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter

import pytest

from tak.ai import (
    AGENTS,
    Agent,
    GreedyAgent,
    LearnerAgent,
    RandomAgent,
    StrategistAgent,
    make_agent,
)
from tak.ai.base import player_headers, position_key
from tak.ai.evaluation import evaluate
from tak.ai.tactics import find_win
from tak.engine import (
    Color,
    Game,
    Move,
    PieceType,
    PlaceMove,
    Square,
    format_move,
    game_from_ptn,
    parse_move,
)

from .conftest import Position, bot_game_positions, random_positions

CORNERS = {Square.from_name(name) for name in ("a1", "e1", "a5", "e5")}

# 4x4, White to move, and d3< is the one move that makes Tinue.
WHITE_TINUE = """[Size "4"]
1. d1 a1 2. c3 c1 3. c4 d2 4. b1 b2 5. c2 b2> 6. d4 2c2< 7. d3 a4 8. a3 b3 9. c2 b3>
"""
# 4x4, Black to move, and d3+ is the one move that makes Tinue.
BLACK_TINUE = """[Size "4"]
1. a1 d1 2. c1 Sc3 3. d2 a2 4. Sb2 a2+ 5. d4 d3 6. b2> b4 7. Sa2 c4 8. b2
"""


def snapshot(game: Game) -> tuple[object, ...]:
    stacks = tuple(game.board.stack(square) for square in game.board.squares())
    return stacks, dict(game.reserves), game.result, game.ply


def wins(game: Game, move: Move) -> bool:
    after = game.copy()
    mover = after.to_move
    after.play(move)
    return after.result is not None and after.result.winner is mover


def safe(game: Game, move: Move) -> bool:
    """Neither loses on the spot nor leaves the opponent a win (checked by the engine)."""
    after = game.copy()
    mover = after.to_move
    after.play(move)
    if after.result is not None:
        return after.result.winner is not mover.opponent
    return after.winning_move(mover.opponent) is None


class TestRegistry:
    def test_four_players_easiest_first(self) -> None:
        assert list(AGENTS) == ["random", "greedy", "strategist", "learner"]
        assert AGENTS["random"] is RandomAgent
        assert AGENTS["greedy"] is GreedyAgent
        assert AGENTS["strategist"] is StrategistAgent
        assert AGENTS["learner"] is LearnerAgent

    def test_make_agent(self) -> None:
        agent = make_agent("greedy", seed=3)
        assert isinstance(agent, GreedyAgent)
        assert agent.name == "Greedy"
        assert agent.description

    def test_unknown_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="choose one of random, greedy, strategist, learner"):
            make_agent("minimax")

    def test_player_headers(self) -> None:
        agent = make_agent("strategist")
        assert player_headers({Color.BLACK: agent}) == {
            "Player1": "Human",
            "Player2": "Strategist (computer)",
        }
        assert player_headers({}) == {}


@pytest.mark.parametrize("key", list(AGENTS))
class TestEveryAgent:
    @pytest.mark.parametrize("size", [3, 4, 5, 6, 7, 8])
    def test_plays_legal_moves_and_leaves_the_game_alone(self, key: str, size: int) -> None:
        agent = make_agent(key, seed=size)
        for game in [Game(size), *random_positions(size, 4, seed=size)]:
            before = snapshot(game)
            move = agent.choose_move(game)
            assert move in game.legal_moves()
            assert snapshot(game) == before

    def test_plays_whole_games(self, key: str) -> None:
        game = Game(4)
        agent = make_agent(key, seed=1)
        while not game.is_over:
            game.play(agent.choose_move(game))
        assert game.result is not None

    def test_the_same_seed_plays_the_same_moves(self, key: str) -> None:
        def moves(seed: int) -> list[Move]:
            game = Game(5)
            agents = [make_agent(key, seed), make_agent(key, seed + 1)]
            for ply in range(12):
                if game.is_over:
                    break
                game.play(agents[ply % 2].choose_move(game))
            return [played.move for played in game.history]

        assert moves(5) == moves(5)

    def test_refuses_a_finished_game(self, key: str) -> None:
        game = Game(3)
        for token in ["a1", "a2", "b2", "a3", "c2"]:
            game.play(parse_move(token))
        with pytest.raises(ValueError, match="over"):
            make_agent(key).choose_move(game)

    def test_is_an_agent(self, key: str) -> None:
        assert isinstance(make_agent(key), Agent)


def after(*moves: str, size: int = 5) -> Game:
    game = Game(size)
    for move in moves:
        game.play(parse_move(move))
    return game


@pytest.mark.parametrize("key", list(AGENTS))
class TestSamePositionSameMove:
    """An agent's move depends only on the position, never on when it's asked."""

    def test_asking_again_gets_the_same_move(self, key: str) -> None:
        agent = make_agent(key)
        for game in [Game(5), after("a1", "e5", "c3"), *random_positions(5, 5, seed=8)]:
            assert len({agent.choose_move(game) for _ in range(5)}) == 1

    def test_taking_moves_back_and_replaying_them(self, key: str) -> None:
        agent = make_agent(key)
        game = Game(5)
        for _ in range(3):  # a flat on the first empty square, then the agent's reply
            empty = next(s for s in game.board.squares() if not game.board.stack(s))
            game.play(PlaceMove(empty))
            game.play(agent.choose_move(game))
        first_time = [played.move for played in game.history]
        for _ in range(4):
            game.undo()
        game.play(first_time[2])
        assert agent.choose_move(game) == first_time[3]
        game.play(first_time[3])
        game.play(first_time[4])
        assert agent.choose_move(game) == first_time[5]

    def test_the_same_position_reached_another_way(self, key: str) -> None:
        agent = make_agent(key)
        one_way = after("a1", "e5", "b2", "c3", "d4")
        other_way = after("a1", "e5", "d4", "c3", "b2")
        assert agent.choose_move(one_way) == agent.choose_move(other_way)

    def test_the_same_seed_plays_the_same_in_every_game(self, key: str) -> None:
        game = after("a1", "e5", "c3")
        assert make_agent(key, seed=42).choose_move(game) == make_agent(key, 42).choose_move(game)


class TestAcrossGames:
    def test_without_a_seed_new_games_can_go_differently(self) -> None:
        game = after("a1", "e5", "c3")
        assert len({RandomAgent().choose_move(game) for _ in range(20)}) > 1
        assert len({GreedyAgent().choose_move(game) for _ in range(20)}) > 1

    def test_moves_dont_depend_on_the_python_process(self) -> None:
        # String seeds are hashed the same way in every run, unlike hash().
        code = (
            "from tak.ai import make_agent; from tak.engine import Game, format_move, parse_move\n"
            "g = Game(5)\n"
            "for m in ['a1', 'e5', 'c3']: g.play(parse_move(m))\n"
            "print(format_move(make_agent('random', seed=7).choose_move(g)))"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
            ).stdout
            for hash_seed in (1, 2)
        }
        assert len(outputs) == 1
        here = format_move(make_agent("random", seed=7).choose_move(after("a1", "e5", "c3")))
        assert outputs == {here + "\n"}

    def test_position_keys(self) -> None:
        assert position_key(after("a1", "e5", "b2", "c3", "d4")) == position_key(
            after("a1", "e5", "d4", "c3", "b2")
        )
        assert position_key(after("a1", "e5", "b2")) != position_key(after("a1", "e5", "Sb2"))
        assert position_key(Game(5)) != position_key(Game(6))


class TestRandom:
    def test_picks_all_sorts_of_moves(self) -> None:
        game = game_from_ptn('[Size "5"]\n1. a1 e5 2. c3 c2')
        chosen = Counter(RandomAgent(seed).choose_move(game) for seed in range(300))
        assert len(chosen) > 40
        kinds = {move.piece for move in chosen if isinstance(move, PlaceMove)}
        assert kinds == {PieceType.FLAT, PieceType.WALL, PieceType.CAP}


@pytest.mark.parametrize("kind", [GreedyAgent, StrategistAgent, LearnerAgent])
class TestTactics:
    """What the greedy player, the strategist and the learner promise: win, block, stay safe."""

    def test_opens_in_a_corner(self, kind: type[Agent]) -> None:
        for seed in range(10):
            game = Game(5)
            move = kind(seed).choose_move(game)
            assert isinstance(move, PlaceMove) and move.square in CORNERS
            game.play(move)
            reply = kind(seed).choose_move(game)
            assert isinstance(reply, PlaceMove) and reply.square in CORNERS - {move.square}

    def test_takes_an_immediate_win(self, kind: type[Agent], position: Position) -> None:
        game = position(5, {"a3": "w", "b3": "w", "c3": "w", "d3": "w", "a1": "b", "b1": "b"})
        for seed in range(5):
            assert wins(game, kind(seed).choose_move(game))

    def test_blocks_the_opponents_win(self, kind: type[Agent], position: Position) -> None:
        # Black threatens d2; White can't win first.
        game = position(4, {"a2": "b", "b2": "b", "c2": "b", "a4": "w", "c4": "w"})
        for seed in range(5):
            move = kind(seed).choose_move(game)
            assert isinstance(move, PlaceMove) and move.square == Square.from_name("d2")

    def test_never_hands_over_a_win_it_could_avoid(self, kind: type[Agent]) -> None:
        positions = [*random_positions(4, 12, seed=3), *random_positions(5, 12, seed=4)]
        positions += bot_game_positions(4, seed=2, white="greedy", black="greedy")
        for number, game in enumerate(positions):
            move = kind(number).choose_move(game)
            if game.winning_move(game.to_move) is not None:
                assert wins(game, move)
            elif any(safe(game, other) for other in game.legal_moves()):
                assert safe(game, move)


class TestGreedy:
    def test_favors_moves_that_gain_squares(self) -> None:
        # Placing a flat or the capstone gains a square; a wall or moving e5 doesn't.
        game = game_from_ptn('[Size "5"]\n1. a1 e5')
        chosen = Counter()
        for seed in range(200):
            move = GreedyAgent(seed).choose_move(game)
            gains = isinstance(move, PlaceMove) and move.piece is not PieceType.WALL
            chosen[gains] += 1
        # Picking uniformly at random would gain a square about 65% of the time.
        assert chosen[True] / 200 > 0.8

    def test_still_varies_its_moves(self) -> None:
        game = game_from_ptn('[Size "5"]\n1. a1 e5')
        assert len({GreedyAgent(seed).choose_move(game) for seed in range(30)}) > 10


class TestStrategist:
    @pytest.mark.parametrize(("ptn", "tinue"), [(WHITE_TINUE, "d3<"), (BLACK_TINUE, "d3+")])
    def test_plays_the_move_that_makes_tinue(self, ptn: str, tinue: str) -> None:
        game = game_from_ptn(ptn)
        assert find_win(game, game.to_move) is None
        for seed in range(3):
            assert StrategistAgent(seed).choose_move(game) == parse_move(tinue)

    def test_otherwise_plays_the_best_looking_safe_move(self) -> None:
        agent = StrategistAgent(seed=0)
        positions = bot_game_positions(5, seed=9, white="greedy", black="greedy")[4:40:4]
        checked = 0
        for game in positions:
            me = game.to_move
            if find_win(game, me) is not None:
                continue
            scores = {}
            for move in game.legal_moves():
                if not safe(game, move):
                    continue
                after = game.copy()
                after.play(move)
                if not after.is_over and after.is_tinue(me):
                    break  # a Tinue position: tested above
                bonus = agent.weights.threat if find_win(after, me) else 0.0
                scores[move] = evaluate(after, me, agent.weights) + bonus
            else:
                if scores:
                    chosen = agent.choose_move(game)
                    assert scores[chosen] == pytest.approx(max(scores.values()))
                    checked += 1
        assert checked >= 5

    def test_prefers_extending_its_road(self, position: Position) -> None:
        # White's a3-b3 row can grow to c3. Black's lone corner stone has two
        # equally short ways to a road, so blocking one of them gains nothing.
        game = position(5, {"a3": "w", "b3": "w", "e1": "b"})
        for seed in range(4):
            assert StrategistAgent(seed).choose_move(game) == parse_move("c3")

    def test_blocks_the_opponents_shortest_road(self, position: Position) -> None:
        # Black's e-file (e1, e5) is the shortest road on the board, and e3 cuts it.
        game = position(5, {"a3": "w", "b3": "w", "e5": "b", "e1": "b"})
        assert StrategistAgent(seed=1).choose_move(game) == parse_move("e3")
