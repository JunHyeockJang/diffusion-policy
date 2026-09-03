import numpy as np
import robosuite as suite


def make_state(obs):
    return np.concatenate(
        [
            obs["object-state"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_gripper_qpos"],
        ]
    )


env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
)

all_observations = []
all_actions = []

num_episode = 100

for episode in range(num_episode):
    obs = env.reset()

    phase = "above"

    grasp_steps = 0
    lift_steps = 0

    episode_observations = []
    episode_actions = []

    state_history = []

    initial_cube_z = obs["cube_pos"][2]

    cube_lifted = False

    for step in range(300):
        eef_pos = obs["robot0_eef_pos"]
        cube_pos = obs["cube_pos"]

        action = np.zeros(env.action_dim)

        if phase == "above":
            target = cube_pos + np.array([0.0, 0.0, 0.10])
            error = target - eef_pos

            action[:3] = np.clip(error * 10.0, -1.0, 1.0)
            action[6] = -1.0

            if np.linalg.norm(error) < 0.02:
                phase = "down"

        elif phase == "down":
            target = cube_pos + np.array([0.0, 0.0, 0.01])
            error = target - eef_pos

            action[:3] = np.clip(error * 10.0, -1.0, 1.0)
            action[6] = -1.0

            if np.linalg.norm(error) < 0.015:
                phase = "grasp"

        elif phase == "grasp":
            action[6] = 1.0

            grasp_steps += 1

            if grasp_steps >= 30:
                phase = "lift"

        elif phase == "lift":
            action[2] = 0.5
            action[6] = 1.0

            lift_steps += 1

            if lift_steps >= 40:
                break

        state = make_state(obs)

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

        episode_observations.append(obs_sequence.copy())
        episode_actions.append(action.copy())

        obs, reward, done, info = env.step(action)

        env.render()

        cube_lifted = obs["cube_pos"][2] > initial_cube_z + 0.05

    if cube_lifted:
        all_observations.extend(episode_observations)
        all_actions.extend(episode_actions)

        print(f"episode {episode}: SUCCESS ({len(episode_observations)} samples)")

    else:
        print(f"episode {episode}: FAILED")

all_observations = np.array(all_observations)
all_actions = np.array(all_actions)

print("observations:", all_observations.shape)
print("actions:", all_actions.shape)

np.savez(
    "lift_dataset_recent_many.npz",
    observations=all_observations,
    actions=all_actions,
)

env.close()
