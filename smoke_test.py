
from eval.arena import play_match

if __name__ == "__main__":
    winner = play_match(
        ["liars_dice.agents.random_agent:RandomAgent", "liars_dice.agents.random_agent:RandomAgent"],
        num_players=2, dice_per_player=2, seed=123, verbose=True
    )
    print("Winner:", winner)
