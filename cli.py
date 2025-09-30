import argparse
from eval.arena_verbose import play_match

def main():
    parser = argparse.ArgumentParser(description="Liar's Dice runner")
    parser.add_argument("--p", nargs="+", required=True, help="Agent paths like liars_dice.agents.random_agent:RandomAgent for each player")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--dice", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    play_match(args.p, num_players=args.players, dice_per_player=args.dice, seed=args.seed)

if __name__ == "__main__":
    main()
