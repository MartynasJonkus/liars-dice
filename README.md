
# Liar's Dice (Python) – Starter Kit

This minimal project gives you:
- Core game engine with public/private observations and legal actions
- Pluggable agent interface
- CLI runner

> Rules: 1s are wild, no palifico/exact calls.

## Quickstart

```bash
# From this folder in VS Code terminal
python -m venv .venv
# Activate the venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
# or macOS/Linux
source .venv/bin/activate

# Install (none required yet, but you can add numpy later)
python -m pip install --upgrade pip

# Run a demo match with two "random" agents (runner chooses random legal actions)
python cli.py --p liars_dice.agents.random_agent:RandomAgent liars_dice.agents.random_agent:RandomAgent --players 2 --verbose
```

## Add your MCTS agent
Create `liars_dice/agents/mcts_belief.py` with a class `MCTSBeliefAgent` that implements:
```python
def select_action(self, obs) -> (str, any): ...
def notify_result(self, obs, info: dict) -> None: ...
```
The runner gives you the **current observation**, and you can call back into a copy of the game for simulations (you'll likely mirror rules in a fast simulator).

## Notes
- The current runner supplies legal actions and falls back to random if an agent returns an illegal action.
- You can evolve the API to pass legal actions into the agent if you prefer.
- Extend the ruleset and metrics as you go.
```

