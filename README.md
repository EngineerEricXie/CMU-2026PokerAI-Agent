# CMU Poker AI Agent

A Python poker-AI project for a custom heads-up Texas Hold'em variant used in a
CMU-style bot competition. The repository contains the game environment,
baseline agents, an API-based match runner, smoke tests, and a Monte Carlo
`PlayerAgent` implementation designed to make fast decisions under a match time
bank.

## Highlights

- Custom 27-card poker variant with a mandatory flop discard round
- Gym-style environment for deterministic local simulation
- FastAPI agent wrapper so bots can be evaluated as independent services
- Baseline agents for regression testing and strategy comparison
- Monte Carlo equity estimation with cached hand evaluation
- Local match runner that records hand-by-hand CSV output
- Pytest coverage for engine behavior and API failure handling

## Game Variant

This project is not standard Texas Hold'em. It models a compact heads-up variant
with a smaller deck and an extra discard decision.

### Deck

The deck has 27 cards:

- Ranks: `2 3 4 5 6 7 8 9 A`
- Suits: diamonds, hearts, spades
- No clubs
- No ten, jack, queen, or king

Cards are encoded as integers from `0` to `26`:

```python
RANKS = "23456789A"
SUITS = "dhs"

rank_index = card_int % len(RANKS)
suit_index = card_int // len(RANKS)
```

### Hand Flow

Each hand follows this sequence:

1. Each player receives 5 private cards.
2. Small blind posts 1 chip; big blind posts 2 chips.
3. Pre-flop betting begins.
4. The flop reveals 3 community cards.
5. Each player must keep 2 of their 5 private cards and discard the other 3.
6. Discarded cards are revealed and removed from the hand.
7. Flop, turn, and river betting continue.
8. The hand ends by fold or showdown.

The final hand uses 2 kept private cards plus the 5-card board.

### Hand Rankings

From strongest to weakest:

1. Straight flush
2. Full house
3. Flush
4. Straight
5. Three of a kind
6. Two pair
7. One pair
8. High card

Four-of-a-kind is impossible because the deck only has three suits. Ace can be
used as a high card or as a low card in straights.

## Repository Layout

```text
.
|-- agents/
|   |-- agent.py          # Abstract FastAPI-backed agent interface
|   |-- test_agents.py    # Fold, call, all-in, and random baselines
|   |-- prob_agent.py     # Monte Carlo probability baseline
|   |-- rl_agent.py       # Optional trained-policy agent loader
|   |-- v10.py            # Experimental strategy snapshot
|   `-- v11.py            # Experimental strategy snapshot
|-- docs/
|   |-- rules.md          # Tournament and game rules
|   |-- game-engine.md    # Environment/action/observation details
|   |-- submission.md     # Bot submission format
|   `-- terminology.md    # Poker terms and engine vocabulary
|-- submission/
|   `-- player.py         # Main PlayerAgent implementation
|-- agent_config.json     # Config for local two-agent matches
|-- agent_test.py         # Submission smoke test against baselines
|-- api_test.py           # API and match-runner tests
|-- engine_test.py        # Engine behavior tests
|-- gym_env.py            # Core PokerEnv implementation
|-- match.py              # API match orchestration
|-- run.py                # Config-driven local match entrypoint
|-- train_rl_agent.py     # Optional policy-gradient training experiment
`-- requirements.txt      # Core runtime and test dependencies
```

## Core Components

### `gym_env.py`

`PokerEnv` is the central game environment. It owns card dealing, street
transitions, valid action calculation, betting rules, discard handling, reward
calculation, and showdown evaluation.

The action enum is:

```python
class ActionType(Enum):
    FOLD = 0
    RAISE = 1
    CHECK = 2
    CALL = 3
    DISCARD = 4
    INVALID = 5
```

Every action is represented as:

```python
(action_type, raise_amount, keep_card_1, keep_card_2)
```

During the discard round, `keep_card_1` and `keep_card_2` are the two card
indices, from `0` to `4`, that the agent keeps. The remaining three cards are
discarded.

### `agents/agent.py`

The base `Agent` class wraps each bot in a small FastAPI server with two
endpoints:

- `GET /get_action`: asks the active bot for an action
- `POST /post_observation`: sends passive observations and terminal rewards

This mirrors the competition setup where bots run as independent services
rather than direct Python objects inside the same process.

### `match.py`

`run_api_match` starts from two agent URLs, plays a fixed number of hands, tracks
bankrolls and time usage, writes a CSV trace, and returns a normalized match
result:

```python
{
    "status": "completed",
    "result": "win",
    "bot0_reward": 100,
    "bot1_reward": -100,
    "bot0_time_used": 1.23,
    "bot1_time_used": 0.42,
}
```

### `submission/player.py`

`PlayerAgent` is the main playable bot. Its strategy combines:

- Monte Carlo equity estimation
- Cached 5-card hand evaluation
- Discard optimization over all 10 possible keep-two combinations
- Pot-odds-aware calling
- Risk control against large bets
- Dynamic simulation counts based on remaining time budget
- A late-match lead lock that reduces variance when the accumulated chip lead is
  large enough

The strategy is intentionally lightweight enough to run under a 1000-hand match
time bank.

## Setup

Python 3.12 is recommended.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Optional RL Dependencies

The main engine and Monte Carlo agent do not require PyTorch. Install the RL
extra only if you want to run the training or policy-loading experiments:

```bash
python -m pip install ".[rl]"
```

## Quick Start

Run a local match using `agent_config.json`:

```bash
python run.py
```

The default config runs:

- `bot0`: `submission.player.PlayerAgent`
- `bot1`: `agents.v11.V11`

Match output is written to `match.csv` by default. CSV files and runtime logs are
ignored by Git.

## Configure A Match

Edit `agent_config.json`:

```json
{
  "bot0": {
    "file_path": "submission.player.PlayerAgent",
    "port": 8000,
    "player_id": "bot0"
  },
  "bot1": {
    "file_path": "agents.test_agents.RandomAgent",
    "port": 8001,
    "player_id": "bot1"
  },
  "match_settings": {
    "csv_output_path": "./match.csv"
  }
}
```

`file_path` should use Python import syntax:

```text
package.module.ClassName
```

## Testing

Run the full test suite:

```bash
python -m pytest
```

Run the submission smoke test against baseline agents:

```bash
python agent_test.py
```

The smoke test validates that `submission.player.PlayerAgent` can be imported,
started as an API server, and complete short matches against baseline opponents.

## Creating A New Agent

Create a class that inherits from `agents.agent.Agent` and implements `act`.

```python
from agents.agent import Agent
from gym_env import PokerEnv


class MyAgent(Agent):
    def __name__(self):
        return "MyAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]
        action_types = PokerEnv.ActionType

        if valid_actions[action_types.DISCARD.value]:
            return (action_types.DISCARD.value, 0, 0, 1)

        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)

        if valid_actions[action_types.CALL.value]:
            return (action_types.CALL.value, 0, 0, 0)

        return (action_types.FOLD.value, 0, 0, 0)
```

Then point `agent_config.json` at the class:

```json
"file_path": "agents.my_agent.MyAgent"
```

## Observation Reference

Each player receives an observation dictionary:

```python
{
    "street": int,
    "acting_agent": int,
    "my_cards": list[int],
    "community_cards": list[int],
    "my_bet": int,
    "my_discarded_cards": list[int],
    "opp_bet": int,
    "opp_discarded_cards": list[int],
    "min_raise": int,
    "max_raise": int,
    "valid_actions": list[int],
    "time_used": float,
    "time_left": float,
    "opp_last_action": str,
    "pot_size": int,
    "blind_position": int
}
```

Hidden or unused card slots are represented as `-1`.

## Development Notes

- Invalid actions are treated as folds by the engine.
- Always check `valid_actions` before returning an action.
- `raise_amount` must be between `min_raise` and `max_raise`.
- During discard, the two keep-card indices must be distinct and between `0`
  and `4`.
- Runtime logs are written under `agent_logs/`.
- Match traces are CSV files and are intentionally ignored.

## Known Limitations

- The environment uses `gym==0.26.2`, which prints a deprecation warning. A
  future cleanup could migrate this project to Gymnasium.
- RL experiments require optional PyTorch installation and are not part of the
  default runtime path.
- The visualizer expects local image assets that are not required for core match
  execution.

## Documentation

Additional details are available in:

- [Tournament rules](docs/rules.md)
- [Game engine deep dive](docs/game-engine.md)
- [Submission guide](docs/submission.md)
- [Terminology](docs/terminology.md)

## License

No open-source license has been declared yet. Until a license is added, the code
is visible for reference but not explicitly licensed for reuse.
