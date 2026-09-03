import numpy as np
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, obs_dim=19, action_dim=7, obs_steps=2, action_steps=16):
        super().__init__()

        self.action_dim = action_dim
        self.action_steps = action_steps

        self.net = nn.Sequential(
            nn.Linear(obs_dim * obs_steps, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * action_steps),
        )

    def forward(self, obs):
        x = obs.flatten(start_dim=1)
        x = self.net(x)

        actions = x.reshape(-1, self.action_steps, self.action_dim)
        return actions


data = np.load("lift_dataset_2obs_16actions.npz")

observations = torch.tensor(
    data["observations"],
    dtype=torch.float32,
)

actions = torch.tensor(
    data["actions"],
    dtype=torch.float32,
)

print("observations:", observations.shape)
print("actions:", actions.shape)

policy = Policy()

optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=1e-3,
)

loss_fn = nn.MSELoss()


for epoch in range(1000):
    pred_actions = policy(observations)

    loss = loss_fn(
        pred_actions,
        actions,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")


torch.save(
    policy.state_dict(),
    "lift_2obs_16actions.pt",
)

print("saved: lift_2obs_16actions.pt")
