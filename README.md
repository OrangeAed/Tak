# Tak

A Python implementation of [Tak](https://en.wikipedia.org/wiki/Tak_(game)), the abstract strategy board game. The project includes a rules-complete game engine, a terminal interface for playing both sides, an optional graphical interface, and AI agents that can play either side.

## Features

- **Complete rules engine** for 3x3 through 8x8 boards: flat stones, walls, capstones, stack movement with carry limits, wall flattening, road wins, flat wins, and double-road handling
- **Terminal play**: play both sides yourself from the command line
- **Optional GUI**: view the board and pieces and make moves graphically
- **AI agents**: algorithms that play as one or both sides, built on the same engine
- **PTN support**: read and write moves and games in Portable Tak Notation
- **Tests** for every layer

The full ruleset this project implements is in [`docs/rules.md`](docs/rules.md).

## Installation

Requires Python 3.12

```bash
git clone <your-repo-url>
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
│       │   ├── moves.py      # PlaceMove, SlideMove, legal move generation
│       │   ├── rules.py      # road detection, flat win, flattening, carry limit
│       │   ├── game.py       # Game state, turn order, apply/undo, result
│       │   └── notation.py   # parse and serialize moves (PTN)
│       ├── cli/              # terminal interface
│       │   ├── __main__.py   # enables `python -m tak.cli`
│       │   ├── render.py     # board -> text
│       │   └── app.py        # input loop, parsing, error messages
│       ├── gui/              # optional graphical interface
│       │   ├── __main__.py
│       │   └── app.py
│       └── ai/               # algorithms that play
│           ├── base.py       # Agent interface: choose_move(game) -> Move
│           ├── random_agent.py
│           ├── evaluation.py # heuristics
│           └── minimax.py    # search
├── tests/                    # mirrors src/tak
│   ├── conftest.py           # shared fixtures (e.g. board builders)
│   ├── engine/
│   ├── cli/
│   ├── ai/
│   └── gui/
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
| `tests/` | The test suite, laid out to mirror `src/tak/` |
| `scripts/` | Standalone helper scripts that are not part of the installed package |

### Packages

**`tak.engine`: the rules.** This is the heart of the project. It contains the board, pieces, legal move generation, win detection, and game state. It does no printing, input, or drawing, so it can be driven by any front end or algorithm.

**`tak.cli`: terminal play.** Renders the board as text and runs an input loop so a person can play both sides. All rules and move validation come from the engine.

**`tak.gui`: graphical play (optional).** Draws the board and pieces and turns clicks into moves. Its dependency is an optional extra, so the rest of the project does not require it.

**`tak.ai`: computer players.** Every AI implements the `Agent` interface in `base.py`: given a game, return a move. Included agents start with a random baseline and build up to heuristic evaluation and minimax search.

## Design principles

- **Dependencies point one way.** `cli`, `gui`, and `ai` depend on `engine`; `engine` depends on none of them. One rules implementation serves every interface.
- **Everything that plays is an agent.** A human at the terminal, a GUI click handler, and an AI are all "something that returns a move for a given game." The game loop does not care who controls which side, so human vs. human, human vs. AI, and AI vs. AI all work with no special cases.
- **Legal moves come from the engine.** The CLI, GUI, and AI all consume the same legal move list, so they can never disagree about what is allowed.
- **`src/` layout.** Keeping the package under `src/` ensures tests run against the installed package rather than whatever happens to be in the working directory.
- **Tests mirror the source tree.** The engine gets the heaviest testing, since road detection, wall flattening, and carry limits are common sources of bugs, and an AI built on a buggy engine learns the bugs.

## Development

```bash
pytest                           # run the tests
ruff check .                     # lint
ruff format .                    # format
mypy src                         # type check
```

CI runs the same checks on every push, so running them locally before committing will save you a failed build.

## Notation

Games and moves use [Portable Tak Notation (PTN)](https://ustak.org/portable-tak-notation/), the community standard. This makes it easy to save and replay games, write test cases from real positions, and interoperate with other Tak tools.

## License

See [`LICENSE`](LICENSE).