import numpy as np
import robosuite as suite
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


env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
)

policy = Policy()
policy.load_state_dict(torch.load("lift_many_recent_policy.pt"))


policy.eval()

obs = env.reset()

state_history = []

for step in range(500):
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

    if len(state_history) == 1:
        obs_sequence = np.stack(
            [
                state_history[0],
                state_history[0],
            ]
        )
    else:
        obs_sequence = np.stack(state_history)

    state_tensor = torch.tensor(obs_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        action = policy(state_tensor)

    action = action.squeeze(0).numpy()
    action = np.clip(action, -1.0, 1.0)

    obs, reward, done, info = env.step(action)

    print(f"step: {step}, action: {action}, reward: {reward}")

    env.render()

env.close()
