from REINFORCE import ReinforceAgent
from CatchGame import CatchGame
from SnakeGame import SnakeGame

## Train reinforce for catch game

catch_game = CatchGame()
snake_game = SnakeGame()
#agent = ReinforceAgent(catch_game, "Catch")
agent = ReinforceAgent(snake_game, "Snake", 4)

agent.train()
