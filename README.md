# Liar's Dice (Python)

> Rules: 1s are wild, no exact calls.

## Quickstart

```bash
# From this folder in terminal
python -m venv .venv
# Activate the venv (Windows PowerShell)
.venv\Scripts\Activate

# Install
python -m pip install --upgrade pip

# Run a demo match
python cli.py --p liars_dice.agents.heuristic:HeuristicAgent liars_dice.agents.ismcts:ISMCTSAgent liars_dice.agents.random:RandomAgent --players 3
```

python -m eval.run_experiments --p liars_dice.agents.heuristic:HeuristicAgent liars_dice.agents.random:RandomAgent liars_dice.agents.ismcts_puct:ISMCTSPUCTAgent --players 3 --dice 5 --num_games 10 --seed 42
