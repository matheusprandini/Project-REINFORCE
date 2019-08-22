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

class ModelSnakeGame(nn.Module):
    def __init__(self, num_actions):
        super(ModelSnakeGame, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc1 = nn.Linear(6400, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
