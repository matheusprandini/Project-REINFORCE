from REINFORCE import ReinforceAgent
from CatchGame import CatchGame

## Train reinforce for catch game

catch_game = CatchGame()
agent = ReinforceAgent(3)

agent.train(catch_game)
