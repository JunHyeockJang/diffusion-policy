import numpy as np
import robosuite as suite
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
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
)

policy = Policy()
policy.load_state_dict(torch.load("lift_basic_policy.pt"))


policy.eval()

obs = env.reset()

for step in range(500):
    state = np.concatenate(
        [
            obs["object-state"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_gripper_qpos"],
        ]
    )

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        action = policy(state_tensor)

    action = action.squeeze(0).numpy()
    action = np.clip(action, -1.0, 1.0)

    obs, reward, done, info = env.step(action)

    print(f"step: {step}, action: {action}, reward: {reward}")

    env.render()

env.close()
