# CMU Poker AI Agent

Python engine, agents, and evaluation scripts for a custom heads-up poker AI
competition variant. The project includes a Gym-style poker environment,
FastAPI-based agent wrappers, baseline agents, a Monte Carlo player agent, and
match tooling for local evaluation.

## Project Layout

```text
.
├── agents/                 # Baseline and experimental agents
├── docs/                   # Competition rules and engine notes
├── submission/             # Public PlayerAgent implementation
├── gym_env.py              # Core poker environment
├── match.py                # API match runner
├── run.py                  # Config-driven local match entrypoint
├── agent_test.py           # Submission smoke test against baseline agents
└── engine_test.py          # Engine behavior tests
```

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

For RL training experiments, also install the optional PyTorch dependency:

```bash
python -m pip install ".[rl]"
```

## Run A Match

Edit `agent_config.json` to choose the two agent classes and ports, then run:

```bash
python run.py
```

By default, `bot0` uses `submission.player.PlayerAgent` and `bot1` uses
`agents.v11.V11`.

## Test

Run the engine and API tests:

```bash
python -m pytest
```

Run the submission smoke test against baseline agents:

```bash
python agent_test.py
```

## Notes

- Match CSV output and runtime logs are ignored by Git.
- RL training artifacts such as `*.pth` are ignored by Git.
- This repository does not currently declare an open-source license.
