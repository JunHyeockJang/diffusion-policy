import numpy as np
import robosuite as suite

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

        t = t.unsqueeze(1).float() / 100.0

        x = torch.cat([obs, noisy_actions, t], dim=1)

        pred_noise = self.net(x)

        return pred_noise.reshape(
            -1,
            self.action_steps,
            self.action_dim,
        )


# Diffusion Schedule
num_diffusion_steps = 100

betas = torch.linspace(
    0.0001,
    0.02,
    num_diffusion_steps,
)

alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def sample_action_chunk(model, obs_tensor):
    x = torch.randn(1, 16, 7)

    for t in reversed(range(num_diffusion_steps)):
        t_tensor = torch.tensor([t])

        with torch.no_grad():
            pred_noise = model(
                obs_tensor,
                x,
                t_tensor,
            )

        alpha = alphas[t]
        alpha_bar = alpha_bars[t]
        beta = betas[t]

        mean = (
            1 / torch.sqrt(alpha) * (x - beta / torch.sqrt(1 - alpha_bar) * pred_noise)
        )

        if t > 0:
            alpha_bar_prev = alpha_bars[t - 1]

            beta_tilde = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)

            noise = torch.randn_like(x)

            x = mean + torch.sqrt(beta_tilde) * noise

        else:
            x = mean

    return x


# Model Load
model = DiffusionPolicy()

model.load_state_dict(
    torch.load(
        "lift_diffusion_policy.pt",
        map_location="cpu",
    )
)

model.eval()


# Environment
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
)

obs = env.reset()

state = np.concatenate(
    [
        obs["object-state"],
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
    ]
)

state_history = [
    state.copy(),
    state.copy(),
]


# Rollout
for plan_step in range(80):
    obs_sequence = np.stack(state_history)

    obs_tensor = torch.tensor(
        obs_sequence,
        dtype=torch.float32,
    ).unsqueeze(0)

    action_chunk = sample_action_chunk(
        model,
        obs_tensor,
    )

    action_chunk = action_chunk.squeeze(0).numpy()

    action_chunk = np.clip(
        action_chunk,
        -1.0,
        1.0,
    )

    for action in action_chunk[:8]:
        obs, reward, done, info = env.step(action)

        state = np.concatenate(
            [
                obs["object-state"],
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"],
            ]
        )

        state_history.append(state.copy())

        if len(state_history) > 2:
            state_history.pop(0)

        env.render()

        if done:
            break

    if done:
        break


env.close()
