"""The strongest opponent: wins, blocks, goes for Tinue, and otherwise plays for position."""

from __future__ import annotations

from dataclasses import dataclass

from tak.ai.base import Agent, opening_move
from tak.ai.evaluation import DEFAULT_WEIGHTS, Weights, evaluate
from tak.ai.tactics import find_win, is_tinue
from tak.engine.game import Game
from tak.engine.moves import Move


@dataclass
class _Option:
    """A move considered, and what playing it would lead to."""

    move: Move
    safe: bool  # the opponent can't win straight after it (and it doesn't lose)
    score: float  # how good the position after it looks, from evaluation.evaluate
    win_next: Move | None  # a win we would threaten after it (Tak), if any


class StrategistAgent(Agent):
    """Wins, blocks, plays for Tinue, and otherwise improves its position.

    Each turn, in order:

    1. If a move wins the game right now, play it.
    2. Try every legal move and set aside those after which the opponent
       could win straight away. When the opponent is threatening to win, the
       moves left are the ones that block it. (If nothing blocks it, the game
       is lost whatever it plays, so it plays on as best it can.)
    3. Among the safe moves, if one leaves a threat the opponent can't stop
       (Tinue), play it.
    4. Otherwise play the safe move that leads to the best-looking position,
       as measured by :func:`tak.ai.evaluation.evaluate`: the longest road,
       how close each player is to completing a road, control of the center,
       the number of squares controlled, and the flat count. Making a threat
       (Tak) earns a bonus. Ties are broken at random.

    On the first turn it puts the opponent's stone in a corner.
    """

    key = "strategist"
    name = "Strategist"
    description = "takes a win, blocks yours, plays for Tinue, and otherwise builds roads"

    def __init__(self, seed: int | None = None, weights: Weights = DEFAULT_WEIGHTS) -> None:
        super().__init__(seed)
        self.weights = weights

    def pick(self, game: Game) -> Move:
        if game.in_opening:
            return opening_move(game, self.rng)
        me = game.to_move
        win = find_win(game, me)
        if win is not None:
            return win

        moves = game.legal_moves()
        self.rng.shuffle(moves)  # so ties go to a random one of the best
        threats: list[Move] = []  # opponent wins seen so far, tried first next time
        options = [self._consider(game, move, threats) for move in moves]
        safe = [option for option in options if option.safe]
        if not safe:
            return max(options, key=lambda option: option.score).move  # lost whatever we do

        # Tinue: try the most promising threats first.
        taks = sorted(
            (option for option in safe if option.win_next is not None),
            key=lambda option: option.score,
            reverse=True,
        )
        for option in taks:
            assert option.win_next is not None
            game.play(option.move)
            try:
                if is_tinue(game, me, hints=[option.win_next]):
                    return option.move
            finally:
                game.undo()

        return max(safe, key=lambda option: option.score).move

    def _consider(self, game: Game, move: Move, threats: list[Move]) -> _Option:
        """Play ``move`` on ``game``, look at the result, and take it back."""
        me = game.to_move
        game.play(move)
        try:
            if game.result is not None:
                # Not a win (that was checked first): a draw is safe, a loss isn't.
                drawn = game.result.winner is None
                return _Option(move, drawn, evaluate(game, me, self.weights), None)
            reply = find_win(game, me.opponent, first=threats)
            if reply is not None:
                if reply not in threats:
                    threats.append(reply)
                return _Option(move, False, evaluate(game, me, self.weights), None)
            win_next = find_win(game, me)
            score = evaluate(game, me, self.weights)
            if win_next is not None:
                score += self.weights.threat
            return _Option(move, True, score, win_next)
        finally:
            game.undo()
