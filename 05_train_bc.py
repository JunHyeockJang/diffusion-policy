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


data = np.load("lift_data.npz")

observations = torch.tensor(data["observations"], dtype=torch.float32)
actions = torch.tensor(data["actions"], dtype=torch.float32)

print("Observations shape:", observations.shape)
print("Actions shape:", actions.shape)

policy = Policy()

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(500):
    pred_actions = policy(observations)

    loss = loss_fn(pred_actions, actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(policy.state_dict(), "bc_policy.pt")
print("Policy saved to bc_policy.pt")
