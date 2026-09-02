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

phase = "above"

for step in range(500):
    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]

    action = np.zeros(env.action_dim)

    # 큐브 n지점까지 위로 이동
    if phase == "above":
        target = cube_pos + np.array([0, 0, 0.1])
        error = target - eef_pos

        action[:3] = np.clip(error * 10, -1.0, 1.0)

        if np.linalg.norm(error) < 0.02:
            phase = "down"

    # n지점 부터 큐브를 향해 내려가기
    elif phase == "down":
        target = cube_pos + np.array([0, 0, 0.01])
        error = target - eef_pos

        action[:3] = np.clip(error * 10, -1.0, 1.0)

        if np.linalg.norm(error) < 0.015:
            phase = "grasp"

    # 큐브 잡기
    elif phase == "grasp":
        action[6] = 1.0  # 그리퍼 닫기

        if step % 50 == 0:  # 그리퍼 닫는 데 시간이 걸리므로 몇 단계 기다림
            phase = "lift"

    # 큐브 들어올리기
    elif phase == "lift":
        action[2] = 0.5
        action[6] = 1.0  # 그리퍼 닫기

    state = np.concatenate(
        [
            obs["object-state"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_gripper_qpos"],
        ]
    )

    observations.append(state.copy())
    actions.append(action.copy())

    obs, reward, done, info = env.step(action)

    print(f"step: {step:03d}, phase: {phase:3s}, reward: {reward:.3f}")

    env.render()

observations = np.array(observations)
actions = np.array(actions)

np.savez("lift_dataset.npz", observations=observations, actions=actions)

print(f"observations: {observations}, actions: {actions}")

env.close()
