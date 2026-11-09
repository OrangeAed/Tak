# Move Notation (PTN)

This project reads and writes games in **Portable Tak Notation (PTN)**, the community standard for recording Tak games as text. This page describes the subset and conventions the `tak.engine.notation` module implements.

> PTN was created by Benjamin Wochinski in 2015 and is documented by the US Tak
> Association at <https://ustak.org/portable-tak-notation/>. Tak is the property
> of Patrick Rothfuss & Crab Fragment LLC. This document is an independent
> summary written for this project; refer to the official specification for the
> authoritative definition.

## Overview

A PTN game is plain text made of three kinds of content:

1. **Headers**: `[Key "Value"]` tags describing the game
2. **Moves**: numbered turns, each holding one move per player
3. **Comments**: free text in braces, ignored by the engine

Notation describes *what was done*, not whether it was legal. Parsing turns text into move objects; checking legality is the job of the engine (see [Parsing vs. legality](#parsing-vs-legality)).

## Coordinates

Squares are named by file (column) and rank (row), for example `c4`.

- Files are lowercase letters starting at `a` on the left
- Ranks are numbers starting at `1` at the bottom
- The board is viewed from Player 1's side, so `a1` is Player 1's near-left corner
- On an N×N board the far-right corner is the Nth letter and Nth number, e.g. `e5` on a 5×5

## Placing a piece

```
[piece][square]
```

| Piece | Letter | Notes |
| ----- | ------ | ----- |
| Flat stone | `F` | Optional; assumed when no letter is given |
| Standing stone (wall) | `S` | Required for walls |
| Capstone | `C` | Required for capstones |

Examples: `d2` places a flat on d2, `Sb4` places a wall on b4, `Ce1` places a capstone on e1.

On each player's first turn they place a flat belonging to their *opponent*. The notation for this looks like any other placement; the engine is responsible for assigning the piece's color.

## Moving a stack

```
[count]square direction [drops] [top piece]
```

| Part | Required | Meaning |
| ---- | -------- | ------- |
| `count` | Only if more than 1 | How many pieces to pick up from the top of the stack |
| `square` | Yes | The square the stack starts on |
| `direction` | Yes | One of `<` `>` `+` `-` (see below) |
| `drops` | Only if the move is a single step | Digits, one per square entered, giving how many pieces to leave on each |
| `top piece` | No | `S` or `C` if the top piece being moved is a wall or capstone |

The drop digits must add up to the count. If the whole carried stack goes onto the one adjacent square, the drops can be left off entirely.

### Directions

| Symbol | Direction | Effect on coordinates |
| ------ | --------- | --------------------- |
| `<` | Left | Toward file `a` |
| `>` | Right | Away from file `a` |
| `+` | Up | Away from rank 1 |
| `-` | Down | Toward rank 1 |

Directions are from Player 1's point of view. The arrows `←` `→` `↑` `↓` are accepted as equivalent to `<` `>` `+` `-` when parsing.

### Examples

| Notation | Meaning |
| -------- | ------- |
| `d4-` | Move the top piece of d4 down one square |
| `3c2>` | Move 3 pieces from c2 one square right, dropping all 3 there |
| `3c2>12` | Take 3 pieces from c2 and go right, dropping 1 then 2 |
| `4a1+211` | Take 4 pieces from a1 and go up, dropping 2, then 1, then 1 |
| `c3>*` | Move the piece on c3 right; the `*` marks a wall being flattened (see below) |

## Annotations

Marks can be added to the end of a move. They carry extra information and never change what the move does.

| Mark | Meaning |
| ---- | ------- |
| `*` | A capstone flattened a wall with this move |
| `'` | The move creates a **tak**: a threat to complete a road next turn |
| `''` (or `"`) | A **tinuë**: a threat the opponent cannot stop |
| `!` | A good or surprising move |
| `?` | A doubtful move or blunder |

`!` and `?` can be combined, e.g. `!!`, `??`, `!?`, `?!`.

The `*` mark is the most useful one to record, because it lets a game be reconstructed and stepped backward unambiguously. Parsers should accept all marks; writers should emit `*` at minimum.

## Results

A finished game ends with one of these result tokens after the last move:

| Token | Meaning |
| ----- | ------- |
| `R-0` / `0-R` | Road win for Player 1 / Player 2 |
| `F-0` / `0-F` | Flat win for Player 1 / Player 2 |
| `1-0` / `0-1` | Win for Player 1 / Player 2 by resignation or time |
| `1/2-1/2` | Draw |

The same value is normally stored in a `Result` header.

## Comments

Anything between `{` and `}` is a comment. Comments may appear between moves and are ignored when parsing. A comment cannot contain a closing brace.

## Headers

Headers appear at the top of the file, one per line, in the form `[Key "Value"]`.

| Header | Used for |
| ------ | -------- |
| `Size` | Board size (3 to 8). Needed to interpret the game correctly |
| `Player1`, `Player2` | Player names |
| `Date` | Date played, formatted `YYYY.MM.DD` |
| `Result` | Result token, as in the table above |
| `Event`, `Site`, `Round` | Where and in what context the game was played |
| `TPS` | Starting position, for games that do not begin from an empty board |

Unknown headers are preserved when a game is loaded and saved again, so files from other Tak tools survive a round trip.

## Putting it together

An illustrative fragment (not a complete or verified game):

```
[Event "Example"]
[Date "2026.11.09"]
[Player1 "Alice"]
[Player2 "Bob"]
[Size "5"]

1. a1 e5
2. c3 Sc2
3. Cd3 b3
4. d3- {Capstone steps down}
```

Move numbers are followed by a period. A move pair is separated by a space, and consecutive turns are separated by a space or newline.

## Parsing vs. legality

The notation layer only answers "what move does this text describe?" It does not know the board state, so it cannot say whether the move is allowed. Keeping the two apart makes both easier to test.

- **Parsing** (`notation.py`): turns a token like `3c2>12` into a move object with a start square, direction, pickup count, and drop list. It rejects malformed text, such as drops that do not sum to the count.
- **Validation** (`rules.py`, `game.py`): checks the move against the actual position. Is the stack yours? Is the carry limit respected? Does it run into a wall?

A simple pattern for the two token shapes, applied after trimming annotation marks:

```python
import re

PLACE = re.compile(r"^(?P<piece>[FSC])?(?P<square>[a-h][1-8])$")
SLIDE = re.compile(
    r"^(?P<count>[1-8])?(?P<square>[a-h][1-8])"
    r"(?P<dir>[<>+-])(?P<drops>[1-8]+)?(?P<top>[FSC])?$"
)
```

After matching a slide, check that the drop digits sum to the count (treating a missing count as 1 and missing drops as a single drop of the full count).

## Implementation notes

- Parse leniently, write strictly: accept the arrow alternatives and optional `F`, but always write the shortest standard form.
- Round-trip tests are worth having: parse a PTN file, write it back out, and compare.
- Keep sample games under `tests/data/` and replay them in tests to confirm the engine reaches the recorded result.