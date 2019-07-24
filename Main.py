from REINFORCE import ReinforceAgent
from CatchGame import CatchGame

## Train reinforce for catch game

catch_game = CatchGame()
agent = ReinforceAgent(catch_game, "Catch")

agent.train()
