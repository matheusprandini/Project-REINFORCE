import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ModelCatchGame(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ModelCatchGame, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)