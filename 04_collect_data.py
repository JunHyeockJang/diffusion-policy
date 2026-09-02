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

observations = []
actions = []

obs = env.reset()

for step in range(200):
    state = np.concatenate(
        [
            obs["object-state"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_gripper_qpos"],
        ]
    )

    # cube가 gripper 기준 어느 방향으로 있는지
    direction = obs["gripper_to_cube_pos"]

    action = np.zeros(env.action_dim)

    # cube 방향으로 이동
    action[:3] = np.clip(direction * 10.0, -1.0, 1.0)

    observations.append(state.copy())
    actions.append(action.copy())

    obs, reward, done, info = env.step(action)

    env.render()


observations = np.array(observations)
actions = np.array(actions)

print(f"Observations shape: {observations.shape}, Actions shape: {actions.shape}")

np.savez("lift_data.npz", observations=observations, actions=actions)

env.close()
