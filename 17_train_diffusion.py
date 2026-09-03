import numpy as np
import torch
import torch.nn as nn


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim=19, obs_steps=2, action_dim=7, action_steps=16):
        super().__init__()

        self.action_dim = action_dim
        self.action_steps = action_steps

        input_dim = obs_dim * obs_steps + action_dim * action_steps + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * action_steps),
        )

    def forward(self, obs, noisy_actions, t):
        obs = obs.flatten(start_dim=1)
        noisy_actions = noisy_actions.flatten(start_dim=1)

        # t : [B] => [B, 1]
        t = t.unsqueeze(1).float() / 100.0

        x = torch.cat([obs, noisy_actions, t], dim=1)
        pred_noise = self.net(x)
        pred_noise = pred_noise.reshape(-1, self.action_steps, self.action_dim)

        return pred_noise


data = np.load("lift_dataset_2obs_16actions.npz")

observations = torch.tensor(data["observations"], dtype=torch.float32)
actions = torch.tensor(data["actions"], dtype=torch.float32)

num_diffusion_steps = 100
betas = torch.linspace(0.0001, 0.02, num_diffusion_steps)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

model = DiffusionPolicy()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    batch_size = observations.shape[0]

    # sample마다 random diffusion timestep
    t = torch.randint(0, num_diffusion_steps, (batch_size,))

    # action과 같은 shape의 noise
    noise = torch.randn_like(actions)

    # [B] => [B, 1, 1]
    alpha_bar = alpha_bars[t].reshape(batch_size, 1, 1)

    # action => noisy action
    noisy_actions = torch.sqrt(alpha_bar) * actions + torch.sqrt(1 - alpha_bar) * noise

    pred_noise = model(observations, noisy_actions, t)
    loss = loss_fn(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss.item():.6f}")


torch.save(model.state_dict(), "lift_diffusion_policy.pt")

print("====done====")
