import numpy as np
import robosuite as suite
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


env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
)

policy = Policy()
policy.load_state_dict(torch.load("lift_2obs_16actions.pt"))


policy.eval()

obs = env.reset()

state_history = []

# 초기 state
state = np.concatenate(
    [
        obs["object-state"],
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
    ]
)

# 처음에는 같은 state 2개
state_history = [
    state.copy(),
    state.copy(),
]

for step in range(80):
    obs_sequence = np.stack(state_history)

    state_tensor = torch.tensor(obs_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        action_chunk = policy(state_tensor)

    action_chunk = action_chunk.squeeze(0).numpy()
    action_chunk = np.clip(action_chunk, -1.0, 1.0)

    # Receding Horizon Control
    # 2개 관측 → 16개 예측 → 8개 실행 → 다시 관측해서 계획!!
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
