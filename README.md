# Tak

A Python implementation of [Tak](https://en.wikipedia.org/wiki/Tak_(game)), the abstract strategy board game. The project includes a rules-complete game engine, a terminal interface, an optional graphical interface, and four computer opponents to play against in either one, one of which taught itself to play.

## Features

- **Complete rules engine** for 3x3 through 8x8 boards: flat stones, walls, capstones, stack movement with carry limits, wall flattening, road wins, flat wins, and double-road handling
- **Terminal play**: play both sides yourself from the command line
- **Graphical play (optional)**: a 3D pygame view of the board you can turn and tilt, with ranks and files labelled, wooden pieces stacked layer by layer, and both players' reserves; type PTN moves to play both sides
- **Tak and Tinue calls**: after each move, terminal play announces "Tak!" when the move threatens to win next turn, or "Tinue!" when the opponent has no way to stop it
- **PTN support**: read and write moves and games in Portable Tak Notation
- **Saved games**: every finished game, in the terminal or the window, is saved as a PTN file in `saves/`
- **Computer opponents**: four of them, in the terminal or the window (`python -m tak.play`). Three follow fixed, hand-written rules, from a random mover to a strategist that plays for Tinue
- **A self-taught opponent**: the Learner learned how to judge positions by playing thousands of games against itself, starting from nothing. It beats all three hand-written opponents, and you can retrain it from scratch in under 20 minutes (`python scripts/train_learner.py`)
- **Tests** for the engine, the computer opponents, and both interfaces

The full ruleset this project implements is in [`docs/rules.md`](docs/rules.md).

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/OrangeAed/Tak.git
cd Tak
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .                 # core engine and CLI
pip install -e ".[dev]"          # adds pytest, ruff, mypy
pip install -e ".[gui]"          # adds pygame, for the GUI
```

## Usage

```bash
python -m tak.cli                # play in the terminal
python -m tak.cli --size 6       # on another board size (3 to 8; the default is 5)

python -m tak.gui                # play in a window (needs the gui extra)
python -m tak.gui --size 6       # on another board size (3 to 8; the default is 5)
python -m tak.gui --load saves/2026-11-09_14-03-22_5x5_white-road.ptn   # step through a saved game

python -m tak.play               # play against the computer (see below for the options)

python -m tak.gui --eval-bar     # any of these with an evaluation bar (see below)
```

**The evaluation bar.** Add `--eval-bar` to `tak.cli`, `tak.gui`, or `tak.play` to show who is ahead, and by how much, as Learner sees it. In the window, a bar runs down the left of the board, split between White's maple and Black's walnut. The side ahead has its score written at its end, and a mark shows where it would be even. The side of the table you're looking from is at the bottom, so the bar turns over when you press Tab. In the terminal, a column of `b` and `w` cells runs down the left of the board (Black's share at the top, on Black's side), with the score beneath it: positive when White is ahead, negative when Black is. A line under the reserves says the same in words, e.g. `Evaluation: White is ahead (+0.31).` The score runs from -1 (Black winning) to +1 (White winning). It comes from Learner's learned evaluation, with the sure-fire tactics on top: if the player to move can win on the spot, or the player who just moved has Tinue, the bar fills for them and reads `Win`. A finished game shows its result. It's the same number Learner gives under "How it rates its position now" when it explains a move. It takes a few milliseconds per move, whoever you're playing, even in a game between two people.

The window shows the board in 3D. White plays light maple and Black dark walnut; flat stones are low octagonal tiles, walls are tiles standing on edge, and capstones are tall bevelled drums. Stacks are built up layer by layer so you can read them from the side, and each stack of two or more has a tag with its height. Ranks and files are marked on all four sides of the frame.

**Moving by clicking.** Click an empty square to place a flat stone. For a wall or capstone, pick it under "Place" in the panel first, or Shift+click (wall) or Ctrl+click (capstone). To move a stack, click it to lift it: as many stones as the carry limit allows come up. Click it again to carry one fewer, then click a neighbouring square to drop a stone there. Click that square again to drop another, or the next square along to move on. The move is played when the last stone is dropped. While a stack is lifted, its square is blue, and the squares you can click next are green. Only legal moves can be made this way, since every step is checked against the engine's list of legal moves. Moves can still be typed in PTN too.

**Stepping through a game.** The buttons under the piece chooser go to the first move, back one, forward one, and to the last move. The last move played is tinted yellow on the board. This works for the game you're playing and for a saved game opened with `--load`, which opens at the start. Playing a move part-way through a game replaces the moves after it. For a loaded game the window asks first (press Enter to go ahead); the file itself is never changed, and if you play on to a finish, the new game is saved as a new file.

The window can be resized, down to a minimum of 860×600.

| Control | Action |
| ------- | ------ |
| Click | Place a piece, or lift a stack and drop it |
| Shift+click / Ctrl+click | Place a wall / a capstone |
| Right-click or Esc | Cancel a lifted stack (Esc also clears the input line) |
| Type a PTN move, then Enter | Play the move |
| Drag with the mouse, or hold the arrow keys | Turn (left/right) and tilt (up/down) the board |
| Mouse wheel | Zoom |
| Tab / Home | Look from the other player's side / reset the view |
| Ctrl+Left / Ctrl+Right (or Ctrl+Z / Ctrl+Y) | Step back / forward one move; hold to keep stepping |
| Ctrl+Home / Ctrl+End | Go to the first / last move |
| Ctrl+Q | Quit |

Held keys repeat, so holding Backspace deletes until you let go.

## Playing against the computer

```bash
python -m tak.play                                  # against Greedy, as White, on 5x5, in the terminal
python -m tak.play --bot learner --color random     # the strongest opponent; a coin toss for your color
python -m tak.play -b random -c black -s 4 --gui    # in the window
```

| Option | Choices | Default |
| ------ | ------- | ------- |
| `-b`, `--bot` | `random`, `greedy`, `strategist`, `learner` (easiest to hardest) | `greedy` |
| `-c`, `--color` | `white`, `black`, or `random`: the color *you* play. White moves first | `white` |
| `-s`, `--size` | 3 to 8 | 5 |
| `--gui` | play in the 3D window instead of the terminal (needs the gui extra) | terminal |
| `--eval-bar` | show the evaluation bar on the left (see [Usage](#usage)) | off |
| `--seed N` | fix the computer's choices (and a random color) across games, see below | |

`--saves-dir DIR` and `--no-save` work as for the other commands. The same game can be started from Python:

```python
from tak.play import play_against_bot

play_against_bot(bot="strategist", color="random", size=6)  # in the terminal
play_against_bot(bot="greedy", color="black", gui=True)  # in the window
```

**The opponents.** Random, Greedy, and Strategist are ordinary code following fixed rules. Learner plays by an evaluation it learned by playing against itself (see [Training the learner](#training-the-learner)), but it doesn't learn during a game against you: it always plays with the weights it was trained to. **The same position always gets the same move.** Chance comes into Random's moves, Greedy's weighted picks, and how Strategist and Learner break ties. Each choice is drawn from the position itself, mixed with a seed the bot picks at the start of the game. Within a game, taking moves back and playing them again (or reaching a position by another order of moves) always brings the same reply. A new game gets a new seed, so Random and Greedy play differently from game to game. With `--seed N`, every game uses that seed, so the bot always plays the same move in the same position.

- **Random** plays a legal move chosen at random.
- **Greedy** takes a win if one is available, and otherwise blocks your win if you have one. Failing both, it picks a move at random, weighted heavily (4× per square) toward moves that gain squares topped by its flat stones or capstones. It never plays a move that lets you win next turn when another move avoids that.
- **Strategist** also takes a win and blocks yours. Next it looks for a move that makes Tinue, a threat you can't stop, and plays it. Otherwise it plays the move that leaves the best position by a weighted score. The score counts the length of its longest road against yours, how close each of you is to completing a road, control of the center, the number of squares controlled, and the flat count (which matters more as the game nears its end). Making a threat (Tak) earns a bonus. Like Greedy, it never hands you a win it could avoid.
- **Learner** uses the same sure-fire tactics as Strategist (take a win, block yours, play for Tinue), plus one more: it passes over moves that would let you reply with Tinue. Every other choice, which is most of them, is made by an evaluation it learned by playing against itself. That evaluation weighs 17 simple measurements of the board (flat count, road progress, stacks, reserves, and so on) with weights no one picked by hand. See [Training the learner](#training-the-learner). In the window, **Why this move?** explains any of its moves.

On the first turn Greedy, Strategist and Learner put your stone in a corner, the usual choice. In matches (see `scripts/play_match.py` below), Greedy beats Random almost every game, Strategist beats Greedy almost every game on every board size, and Learner beats all three (see [How well it plays](#how-well-it-plays)). Strategist and Learner think longest. On 5x5 they typically answer in under a tenth of a second, with the odd complicated position taking a second or so. On 7x7 and 8x8 they average a tenth to a quarter of a second, with the odd move taking a second or two. Learner takes about one and a half to two times as long as Strategist.

**In the terminal**, the computer's moves are announced as it makes them (`Black (Strategist) plays c3.`), followed by "Tak!" or "Tinue!" when they threaten. `undo` takes back your last move along with the computer's reply, so it's your turn again.

**In the window**, the computer moves by itself when its turn comes; the panel says who is playing whom and shows when the computer is thinking. The view starts from your side of the board. It thinks in the background, so you can still turn the board or step through the game meanwhile. Stepping through the game works exactly as it does in any other game: one move at a time, with the same buttons and keys. The computer only moves by itself at the latest position, so it waits while you look back. To take back a move, step back to your turn and play something else. As in any game, that replaces the later moves, and the computer answers your new move. If you stop on one of the computer's earlier turns, the panel says so. Step forward to see its move, which is the move it would make there again, or step back to your own turn.

**Asking Learner why.** When you play Learner, the panel has a **Why this move?** button just above the step buttons. It stays in the same place all game, and it is greyed out while the last move is your own. After Learner moves, click it and a card over the board explains the move in a few sentences. It gives the main reason (it won, blocked your threat, made Tinue, or played the move its learned evaluation liked best) and mentions any better-looking move it passed over because you could have answered it with Tinue. For a quiet move, it also lists up to three things the move gained, as its learned weights see it: the flat count, blocking your road, building its own, stacks, the center, and so on. It closes with how Learner rates its position. The card sits at the top or bottom of the board view, whichever keeps clear of the squares the move touched. Click the card, click **Hide the explanation**, or press Esc to close it; it also closes once the game moves on. Stepping back to an earlier Learner move and clicking **Why this move?** explains that one. The explanation comes from Learner's record of how it actually chose the move, not from a guess made afterwards.

Saved games against the computer record who played which side in PTN's `Player1` (White) and `Player2` (Black) headers.

### Computer against computer

```bash
python scripts/play_match.py strategist greedy                    # 20 games on 5x5, swapping colors
python scripts/play_match.py greedy random --games 50 --size 6 --seed 1
```

The script prints each result, then the score and how long each player took per move.

## Training the learner

Learner comes with trained weights (`src/tak/ai/learner_weights.json`), so `--bot learner` works straight away. To train it again from scratch:

```bash
python scripts/train_learner.py                   # about 10 minutes on 4 cores; never more than 20
python scripts/train_learner.py --check 20        # then play 20 games against each bot
python scripts/train_learner.py --games 300 -o my_weights.json   # quicker and weaker, saved elsewhere
```

By default the new weights replace the ones the game uses. Progress is printed as it goes, and at the end it prints what it learned. `--seed` makes a run repeatable: the same seed gives the same weights, however many CPU cores play the games. `--sizes` picks the board sizes to train on (5x5 by default), `--workers` the number of processes, and `--minutes` (20 by default) stops training early if it runs long, keeping what it has learned so far.

**How it learns.** Learner judges a position by 17 simple measurements of the board (`FEATURE_NAMES` in `learning.py`). Most are "ours minus theirs": flats on top, walls, capstones, center control, stones held in stacks, how short of a road each player is, how far the best group reaches, and pieces left in reserve. A few aren't: whether we threaten to win next move, and whether either player is exactly two or three squares short of a road. It multiplies each measurement by a weight, adds them up, and squashes the total into a value from −1 (lost) to +1 (won). Training starts with every weight at zero, so at first every quiet move looks equally good. Then it plays 1,000 games against itself. After each game it uses TD(λ), the method behind TD-Gammon, to nudge the weights so that its opinion of each position moves toward its opinion of the next one, and at the end toward how the game actually went. A few random moves at the start of each game, and the odd random move later, keep the games varied. The step size shrinks as training goes on, and the final weights are the average over the second half of training, which is steadier than any single moment. It never plays the other bots while it learns. Games are played several at a time, one per core, in batches that share the same weights.

**What is learned and what isn't.** Like Strategist, Learner always takes a win, never allows you a win it could prevent, and plays a Tinue when it finds one. It also passes over moves that let you reply with Tinue. These tactics look only a move or two ahead and are never wrong, but they say nothing about most moves, and choosing among those is what the learned weights do. With untrained weights, the same tactics win 1 of 40 games against Greedy and none of 40 against Strategist.

### How well it plays

The weights that come with the game were made by `python scripts/train_learner.py` with its defaults: 1,000 self-play games on 5x5, which took 11.4 minutes on a 4-core machine. Each match below swaps colors every game:

| Board | vs Random | vs Greedy | vs Strategist |
| ----- | --------- | --------- | ------------- |
| 3x3 | 40–0 | 33–2, 5 draws | 30–10 |
| 4x4 | 40–0 | 40–0 | 40–0 |
| 5x5 | 100–0 | 100–0 | 99–1 |
| 6x6 | 40–0 | 40–0 | 39–1 |
| 7x7 | 20–0 | 20–0 | 20–0 |
| 8x8 | 12–0 | 12–0 | 23–9 |

It only trained on 5x5, but the measurements are scaled by board size, so the weights carry over. The 3x3 board favors whoever moves first, and 8x8 is where Strategist does best against it. Training isn't a lucky draw, either: three more runs with other seeds beat Strategist 58–2, 55–5 and 60–0 on 5x5. To run your own matches, use `python scripts/play_match.py learner strategist --games 100`, for example.

## Saved games

When a game ends, in the terminal or in the window, it is saved automatically as a new PTN file in the project's `saves/` folder, and the program says where:

```
Game over, White wins with a road: a2 -> b2 -> c2
Result: R-0
Game saved to saves/2026-11-09_14-03-22_3x3_white-road.ptn
```

Files are named after when the game finished, the board size, and the outcome (`white-road`, `black-flats`, `draw`, ...). An existing save is never overwritten: a second game in the same second gets `-2` added to its name. In the window you can still step back after the game ends; if you then play on to a different finish, its file is updated with the new ending rather than a second file being made. Stepping through a game opened with `--load` never saves it again; only a new ending played from it is saved, as a new file. Games you quit before the end are not saved.

The files are standard PTN with `Size`, `Date`, `Time`, and `Result` headers, so they load back into the engine (or any other Tak tool):

```python
from pathlib import Path
from tak.engine import game_from_ptn

game = game_from_ptn(Path("saves/2026-11-09_14-03-22_3x3_white-road.ptn").read_text())
```

`python -m tak.cli`, `python -m tak.gui`, and `python -m tak.play` all accept `--saves-dir DIR` to save somewhere else and `--no-save` to not save at all. The `TAK_SAVES_DIR` environment variable also changes the folder (the test suite uses it so tests never write into `saves/`).

## Tak and Tinue

In terminal play, when a move threatens to win on the mover's next turn, the game calls it, the way a player would say "check" in chess. The call goes on its own line, right before the line saying whose turn it is:

```
White: 7 stones   Black: 8 stones
Tak!
Turn 3: Black to move.
```

- **Tak!** means that if the player who just moved could move again right away, at least one legal move would win them the game. Any win condition counts: a road (including completing both roads at once, which the mover wins), or a flat win by placing their last piece or filling the board while ahead.
- **Tinue!** replaces "Tak!" when the threat can't be stopped: whatever the opponent replies, the threatening player can still win on their next move. A reply that wins or draws the game for the opponent stops the threat. A reply that hands the threatening player the win on the spot does not, and neither does a reply that blocks every current threat but opens a new one.

The call is made once, straight after the move. It isn't repeated when the board is shown again after an illegal move, a command, or a blank line. Undoing the move removes it, and playing the move again repeats it. The `ptn` command doesn't add PTN's `'` (tak) and `''` (tinue) marks, though they are accepted when reading moves.

The same checks are available from code, for any player regardless of whose turn it is:

```python
from tak.engine import Color, Game, parse_move

game = Game(3)
for move in "a1 c3 b1 b3 b2".split():
    game.play(parse_move(move))

game.threatens_win(Color.WHITE)  # True: c2 would complete a road
game.winning_move(Color.WHITE)  # the move placing a flat on c2 (None if no move wins)
game.is_tinue(Color.WHITE)  # False: Black can block c2
```

Spotting Tinue means trying every reply the opponent could make, so it costs more than spotting Tak. In typical games both checks together take a few milliseconds per move. Contrived 8x8 positions, where the opponent has thousands of possible replies, can take up to about a second.

## Project structure

```
Tak/
├── LICENSE
├── README.md
├── pyproject.toml            # packaging, dependencies, tool config (pytest, ruff, mypy)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml            # runs tests, lint, and type checks on every push
├── docs/
│   ├── rules.md              # the Tak rules this project implements
│   └── notation.md           # the move notation (PTN) this project reads and writes
├── src/
│   └── tak/
│       ├── __init__.py
│       ├── saves.py          # save finished games as PTN files in saves/ (shared by cli and gui)
│       ├── engine/           # game simulation (pure logic, no I/O)
│       │   ├── pieces.py     # PieceType, Color, Piece
│       │   ├── board.py      # Board, stacks, size and reserve config table
│       │   ├── moves.py      # PlaceMove, SlideMove, legal move generation (for either player)
│       │   ├── rules.py      # road detection, flat win, flattening, carry limit
│       │   ├── game.py       # Game state, turn order, apply/undo, result, Tak/Tinue detection
│       │   └── notation.py   # parse and serialize moves (PTN)
│       ├── cli/              # terminal interface
│       │   ├── __main__.py   # enables `python -m tak.cli`
│       │   ├── render.py     # board and status -> text, including the Tak!/Tinue! line
│       │   └── app.py        # input loop, parsing, error messages, Tak/Tinue calls
│       ├── gui/              # optional graphical interface (pygame)
│       │   ├── __main__.py   # enables `python -m tak.gui`
│       │   ├── scene.py      # small 3D renderer: camera, projection, meshes, draw order, picking
│       │   ├── render.py     # game state -> frame: 3D board, labels, stacks, panel and buttons
│       │   ├── clicks.py     # click to move: clicks -> legal moves (no drawing)
│       │   ├── session.py    # the game's timeline: stepping, loading, confirming, saving
│       │   ├── computer.py   # the computer's move, worked out in a background thread
│       │   └── app.py        # window, event handling (Controller), command-line options
│       ├── play.py           # play against the computer: `python -m tak.play`, play_against_bot()
│       └── ai/               # computer players
│           ├── base.py       # Agent interface: choose_move(game) -> Move
│           ├── random_agent.py      # Random: any legal move
│           ├── greedy_agent.py      # Greedy: win, block, otherwise gain squares
│           ├── strategist_agent.py  # Strategist: win, block, Tinue, otherwise best position
│           ├── learner_agent.py     # Learner: the same tactics, otherwise its learned evaluation
│           ├── tactics.py    # fast one-move look-ahead: immediate wins, safe moves, Tinue
│           ├── evaluation.py # scoring a position: roads, center, squares, flats (hand-tuned)
│           ├── learning.py   # the Learner's features and learned evaluation
│           ├── explain.py    # plain-words explanations of the Learner's moves ("Why this move?")
│           ├── assess.py     # who is ahead, as the Learner sees it: the evaluation bar
│           ├── learner_weights.json # the Learner's trained weights
│           ├── training.py   # teaching the Learner: TD(lambda) learning from self-play
│           └── minimax.py    # deeper search (planned; placeholder)
├── tests/                    # mirrors src/tak, plus threats/ for Tak and Tinue
│   ├── conftest.py           # shared fixtures (e.g. board builders); keeps saves out of saves/
│   ├── test_saves.py         # file names, never overwriting, round trips, save options
│   ├── test_play.py          # playing against the computer: options, whole games, the window
│   ├── engine/
│   ├── cli/
│   ├── ai/                   # computer players, with the fast look-ahead checked against the engine;
│   │                         #   test_learner.py covers the Learner's features, search and training,
│   │                         #   test_explain.py its explanations, test_assess.py the evaluation bar
│   ├── gui/                  # clicking, stepping, picking, drawing, controls (headless pygame)
│   └── threats/              # Tak and Tinue, from the engine through to the terminal
│       ├── conftest.py                   # position builder, scripted terminal sessions
│       ├── test_regular_moves.py         # move generation for either player
│       ├── test_tak_detection.py         # threats to win next turn
│       ├── test_tinue_detection.py       # threats that can't be stopped
│       ├── test_threat_announcements.py  # "Tak!" and "Tinue!" in the terminal
│       └── test_road_search.py           # road search, which the checks run constantly
├── scripts/
│   ├── play_match.py         # pit two computer players against each other
│   └── train_learner.py      # train the Learner by self-play, from scratch
└── saves/                    # local saved games and match output (contents ignored by git)
```

### Top-level files and folders

| Path | Purpose |
| ---- | ------- |
| `pyproject.toml` | Single source of truth for packaging, dependencies (including the optional `gui` and `dev` extras), and tool configuration for pytest, ruff, and mypy |
| `.github/workflows/ci.yml` | Continuous integration: runs the test suite, linter, and type checker on every push and pull request |
| `docs/` | Project documentation: the rules specification and the move notation |
| `src/tak/` | The installable package; everything below is importable as `tak.<subpackage>` |
| `tests/` | The test suite, laid out to mirror `src/tak/`, plus `tests/threats/` for Tak and Tinue |
| `scripts/` | Standalone helper scripts that are not part of the installed package |
| `saves/` | Local saved games and match output. Only the `.gitkeep` placeholder is committed |

### Packages

**`tak.engine`: the rules.** This is the heart of the project. It contains the board, pieces, legal move generation, win detection, game state, and threat detection: whether a player threatens to win next turn (Tak), and whether that threat can be stopped (Tinue). It does no printing, input, or drawing, so it can be driven by any front end or algorithm.

**`tak.saves`: saved games.** Writes a finished game as a new PTN file in `saves/` and provides the `--saves-dir` and `--no-save` options. It does file I/O, so it sits beside the engine rather than in it, and both front ends use it.

**`tak.cli`: terminal play.** Renders the board as text and runs an input loop so a person can play both sides, or one side against the computer. After a threatening move it announces "Tak!" or "Tinue!". All rules, move validation, and threat detection come from the engine.

**`tak.gui`: graphical play (optional).** Draws the board in 3D in a pygame window, with rank and file labels, every stack built up piece by piece, and both players' reserves; the view can be turned, tilted, and zoomed. pygame only draws 2D shapes, so `scene.py` does the 3D itself (perspective projection, shading, and back-to-front painting), with no other dependencies. Moves are made by clicking (checked step by step against the engine's legal moves) or typed in PTN, and any game, including a saved one, can be stepped through. Its dependency is an optional extra (`pip install -e ".[gui]"`), so the rest of the project does not require it. Start it with `python -m tak.gui`, or `python -m tak.play --gui` to play the computer; the computer thinks in a background thread (`computer.py`) so the window stays responsive.

**`tak.ai`: computer players.** Every computer player implements the `Agent` interface in `base.py`: given a game, return a move, without changing the game (agents think on a copy, `Game.copy()`). The four players are described under [Playing against the computer](#playing-against-the-computer). The first three are built from two modules of plain, hand-tuned code. `tactics.py` answers "can this player win right now?", "does this move hand the opponent a win?" and "is this Tinue?". It gives the same answers as the engine's own checks but several times faster, which matters because the stronger players ask after every move they consider. Road wins by placement are read off the connected groups of road pieces. A slide is only tried one by one when the line it moves along could possibly complete a road. Near the end, where flat wins are possible, it defers to the engine. `evaluation.py` scores positions for the Strategist. The Learner uses `tactics.py` too, with its own faster search for wins near the end of the game (checked against the engine in the tests), and scores positions with `learning.py`, whose weights come from `training.py` (see [Training the learner](#training-the-learner)). `explain.py` turns the Learner's record of how it chose a move into a short explanation, for the window's **Why this move?** button, and `assess.py` reads a position for the evaluation bar (`--eval-bar`). Deeper search (`minimax.py`) is still a placeholder.

**`tak.play`: playing the computer.** The `python -m tak.play` command and the `play_against_bot()` function: choose the opponent, your color, the board size, and the terminal or the window.

## Design principles

- **Dependencies point one way.** `cli`, `gui`, and `ai` depend on `engine`; `engine` depends on none of them. `cli` and `gui` use `ai`, and `play` sits on top of them all. One rules implementation serves every interface.
- **Everything that plays is an agent.** The goal is for a human at the terminal, a GUI click handler, and an AI to all be "something that returns a move for a given game", so that the game loop does not care who controls which side and human vs. human, human vs. AI, and AI vs. AI all work with no special cases. The terminal loop and the window take an `Agent` for either side (or both); sides without one are played by the person.
- **Legal moves come from the engine.** The CLI checks every move with the engine, and the GUI and the computer players use the same legal move list, so they can never disagree about what is allowed.
- **`src/` layout.** Keeping the package under `src/` ensures tests run against the installed package rather than whatever happens to be in the working directory.
- **Tests mirror the source tree.** Each package gets a matching folder under `tests/`. The GUI tests run pygame headless on SDL's dummy video driver, so they need no display and are skipped entirely if pygame is not installed. The engine gets the heaviest testing, since road detection, wall flattening, and carry limits are common sources of bugs, and a computer player built on a buggy engine plays the bugs. Tak and Tinue span the engine and the terminal, so their tests live together in `tests/threats/`. Alongside hand-built positions, they check the results against brute force that plays out every move with the real engine.

## Development

```bash
pytest                           # run the tests
pytest tests/threats             # just the Tak and Tinue tests
ruff check .                     # lint
ruff format .                    # format
mypy src                         # type check
```

CI (`.github/workflows/ci.yml`) runs the tests, lint, and type check on Python 3.10 through 3.14 on every push and pull request, so running them locally before committing will save you a failed build.

## Notation

Games and moves use [Portable Tak Notation (PTN)](https://ustak.org/portable-tak-notation/), the community standard. This makes it easy to save and replay games, write test cases from real positions, and interoperate with other Tak tools. [`docs/notation.md`](docs/notation.md) describes the subset and conventions this project implements.

## License

See [`LICENSE`](LICENSE).