import numpy as np
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, obs_dim=19, action_dim=7, obs_steps=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim * obs_steps, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

    def forward(self, obs):
        x = obs.flatten(start_dim=1)
        return self.net(x)


data = np.load("lift_dataset_recent_many.npz")

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
    "lift_many_recent_policy.pt",
)

print("saved: lift_many_recent_policy.pt")
