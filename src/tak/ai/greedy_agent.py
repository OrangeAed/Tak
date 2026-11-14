"""The middle opponent: wins when it can, blocks when it must, and grabs squares."""

from __future__ import annotations

from tak.ai.base import Agent, opening_move
from tak.ai.tactics import find_win, loses_or_allows_win, road_cells, stacks_of
from tak.engine.board import Board
from tak.engine.game import Game
from tak.engine.moves import Move
from tak.engine.pieces import Color


def controlled(board: Board, color: Color) -> int:
    """Squares with one of ``color``'s flat stones or capstones on top."""
    return sum(road_cells(stacks_of(board), color))


class GreedyAgent(Agent):
    """Takes an immediate win, stops the opponent's, and otherwise grabs squares.

    Each turn, in order:

    1. If a move wins the game right now, play it.
    2. Otherwise pick a move at random, favoring moves that gain squares
       topped by its flat stones or capstones: each square a move gains makes
       it :attr:`bias` times likelier to be picked (and each square lost, that
       much less likely). This is a bias, not a rule, so it doesn't always
       play the same way.
    3. Before playing the move picked, check that the opponent couldn't win
       straight after it. If they could, set it aside and pick again. So when
       the opponent threatens to win, it blocks (if any move can), and it
       never opens up a win for the opponent when it has a choice.

    On the first turn it puts the opponent's stone in a corner.
    """

    key = "greedy"
    name = "Greedy"
    description = "takes a win, blocks yours, and otherwise grabs squares"

    #: Each square a move gains makes it this many times likelier to be picked.
    bias = 4.0

    def pick(self, game: Game) -> Move:
        if game.in_opening:
            return opening_move(game, self.rng)
        me = game.to_move
        win = find_win(game, me)
        if win is not None:
            return win

        moves = game.legal_moves()
        before = controlled(game.board, me)
        weights = []
        for move in moves:
            game.play(move)
            gain = controlled(game.board, me) - before
            game.undo()
            weights.append(self.bias**gain)

        threats: list[Move] = []  # opponent wins seen so far, tried first next time
        options = list(range(len(moves)))
        while options:
            choice = self.rng.choices(options, [weights[i] for i in options])[0]
            if not loses_or_allows_win(game, moves[choice], threats):
                return moves[choice]
            options.remove(choice)
        # Whatever we play, the opponent can win; pick as usual.
        return self.rng.choices(moves, weights)[0]
