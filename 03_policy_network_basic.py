import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, obs_dim=19, obs_steps=2, action_dim=7):
        super().__init__()

        input_dim = obs_dim * obs_steps

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs_sequence):
        x = obs_sequence.flatten(start_dim=1)
        action = self.net(x)

        return action


policy = Policy()

# batch size, 최근 observation 2개, state dimension
obs_sequence = torch.randn(1, 2, 19)

action = policy(obs_sequence)

print(f"obs: {obs_sequence.shape}, action: {action.shape} and {action}")
