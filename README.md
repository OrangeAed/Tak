# Tak

A Python implementation of [Tak](https://en.wikipedia.org/wiki/Tak_(game)), the abstract strategy board game. The project includes a rules-complete game engine, a terminal interface for playing both sides, an optional graphical interface, and AI agents that can play either side.

## Features

- **Complete rules engine** for 3x3 through 8x8 boards: flat stones, walls, capstones, stack movement with carry limits, wall flattening, road wins, flat wins, and double-road handling
- **Terminal play**: play both sides yourself from the command line
- **Tak and Tinue calls**: after each move, terminal play announces "Tak!" when the move threatens to win next turn, or "Tinue!" when the opponent has no way to stop it
- **Optional GUI**: view the board and pieces and make moves graphically
- **AI agents**: algorithms that play as one or both sides, built on the same engine
- **PTN support**: read and write moves and games in Portable Tak Notation
- **Tests** for every layer

The full ruleset this project implements is in [`docs/rules.md`](docs/rules.md).

## Installation

Requires Python 3.12

```bash
git clone https://github.com/OrangeAed/Tak.git
cd tak
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .                 # core engine, CLI, and AI
pip install -e ".[dev]"          # adds pytest, ruff, mypy
pip install -e ".[gui]"          # adds the GUI dependency
```

## Usage

```bash
python -m tak.cli                # play in the terminal
python -m tak.gui                # launch the graphical interface (requires the gui extra)
python scripts/play_match.py     # pit two AI agents against each other
```

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

game.threatens_win(Color.WHITE)   # True: c2 would complete a road
game.winning_move(Color.WHITE)    # the move placing a flat on c2 (None if no move wins)
game.is_tinue(Color.WHITE)        # False: Black can block c2
```

Spotting Tinue means trying every reply the opponent could make, so it costs more than spotting Tak. In typical games both checks together take a few milliseconds per move. Contrived 8x8 positions, where the opponent has thousands of possible replies, can take up to about a second.

## Project structure

```
tak/
├── LICENSE
├── README.md
├── pyproject.toml            # packaging, dependencies, tool config (pytest, ruff, mypy)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml            # runs tests and lint on every push
├── docs/
│   └── rules.md              # the Tak rules this project implements
├── src/
│   └── tak/
│       ├── __init__.py
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
│       ├── gui/              # optional graphical interface
│       │   ├── __main__.py
│       │   └── app.py
│       └── ai/               # algorithms that play
│           ├── base.py       # Agent interface: choose_move(game) -> Move
│           ├── random_agent.py
│           ├── evaluation.py # heuristics
│           └── minimax.py    # search
├── tests/                    # mirrors src/tak, plus threats/ for Tak and Tinue
│   ├── conftest.py           # shared fixtures (e.g. board builders)
│   ├── engine/
│   ├── cli/
│   ├── ai/
│   ├── gui/
│   └── threats/              # Tak and Tinue, from the engine through to the terminal
│       ├── conftest.py                   # position builder, scripted terminal sessions
│       ├── test_regular_moves.py         # move generation for either player
│       ├── test_tak_detection.py         # threats to win next turn
│       ├── test_tinue_detection.py       # threats that can't be stopped
│       ├── test_threat_announcements.py  # "Tak!" and "Tinue!" in the terminal
│       └── test_road_search.py           # road search, which the checks run constantly
└── scripts/
    └── play_match.py         # pit two agents against each other
```

### Top-level files and folders

| Path | Purpose |
| ---- | ------- |
| `pyproject.toml` | Single source of truth for packaging, dependencies (including the optional `gui` and `dev` extras), and tool configuration for pytest, ruff, and mypy |
| `.github/workflows/ci.yml` | Continuous integration: runs the test suite, linter, and type checker on every push |
| `docs/` | Project documentation, including the rules specification |
| `src/tak/` | The installable package; everything below is importable as `tak.<subpackage>` |
| `tests/` | The test suite, laid out to mirror `src/tak/`, plus `tests/threats/` for Tak and Tinue |
| `scripts/` | Standalone helper scripts that are not part of the installed package |

### Packages

**`tak.engine`: the rules.** This is the heart of the project. It contains the board, pieces, legal move generation, win detection, game state, and threat detection: whether a player threatens to win next turn (Tak), and whether that threat can be stopped (Tinue). It does no printing, input, or drawing, so it can be driven by any front end or algorithm.

**`tak.cli`: terminal play.** Renders the board as text and runs an input loop so a person can play both sides. After a threatening move it announces "Tak!" or "Tinue!". All rules, move validation, and threat detection come from the engine.

**`tak.gui`: graphical play (optional).** Draws the board and pieces and turns clicks into moves. Its dependency is an optional extra, so the rest of the project does not require it.

**`tak.ai`: computer players.** Every AI implements the `Agent` interface in `base.py`: given a game, return a move. Included agents start with a random baseline and build up to heuristic evaluation and minimax search.

## Design principles

- **Dependencies point one way.** `cli`, `gui`, and `ai` depend on `engine`; `engine` depends on none of them. One rules implementation serves every interface.
- **Everything that plays is an agent.** A human at the terminal, a GUI click handler, and an AI are all "something that returns a move for a given game." The game loop does not care who controls which side, so human vs. human, human vs. AI, and AI vs. AI all work with no special cases.
- **Legal moves come from the engine.** The CLI, GUI, and AI all consume the same legal move list, so they can never disagree about what is allowed.
- **`src/` layout.** Keeping the package under `src/` ensures tests run against the installed package rather than whatever happens to be in the working directory.
- **Tests mirror the source tree.** The engine gets the heaviest testing, since road detection, wall flattening, and carry limits are common sources of bugs, and an AI built on a buggy engine learns the bugs. Tak and Tinue span the engine and the terminal, so their tests live together in `tests/threats/`. Alongside hand-built positions, they check the results against brute force that plays out every move with the real engine.

## Development

```bash
pytest                           # run the tests
pytest tests/threats             # just the Tak and Tinue tests
ruff check .                     # lint
ruff format .                    # format
mypy src                         # type check
```

CI runs the same checks on every push, so running them locally before committing will save you a failed build.

## Notation

Games and moves use [Portable Tak Notation (PTN)](https://ustak.org/portable-tak-notation/), the community standard. This makes it easy to save and replay games, write test cases from real positions, and interoperate with other Tak tools.

## License

See [`LICENSE`](LICENSE).