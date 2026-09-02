import numpy as np
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, obs_dim=19, action_dim=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


data = np.load("lift_dataset.npz")

observations = torch.tensor(data["observations"], dtype=torch.float32)
actions = torch.tensor(data["actions"], dtype=torch.float32)

print(f"obs: {observations.shape}, actions: {actions.shape}")

policy = Policy()

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    pred_actions = policy(observations)

    loss = loss_fn(pred_actions, actions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss.item():.6f}")

torch.save(policy.state_dict(), "lift_basic_policy.pt")

print("saved")
