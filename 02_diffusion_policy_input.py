import numpy as np
import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
)

state_history = []

obs = env.reset()

for step in range(200):
    action = np.zeros(env.action_dim)

    action[0] = 0.1

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

    if len(state_history) == 2:
        obs_sequence = np.stack(state_history)
        print(f"state: {state.shape}, Observation sequence shape: {obs_sequence.shape}")

    env.render()

env.close()
