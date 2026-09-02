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

obs = env.reset()

print(f"Observation keys: {obs.keys()}")

print("======Observations======")

low, high = env.action_spec

print(f"action dim: {env.action_dim}")
print(f"action low: {low}")
print(f"action high: {high}")

for step in range(100):
    action = np.zeros(env.action_dim)

    # x, 실제론 delta scaling이 적용되므로 0.3이면 0.3m 이동하는게 아니라 0.3 * delta_scaling만큼 이동한다.
    action[0] = 0.3

    obs, reward, done, info = env.step(action)

    print(f"step: {step}, robot pos: {obs['robot0_eef_pos']}")

    env.render()

env.close()

"""
          environment
              │
              │ obs
              ▼
          ┌────────┐
          │ Policy │
          └────────┘
              │
              │ action
              ▼
          environment

          action = policy(obs)
"""

"""
        obs?

        robot0_joint_pos         => 로봇 관절 위치
        robot0_joint_pos_cos
        robot0_joint_pos_sin     
        robot0_joint_vel         => 로봇 관절 속도

        robot0_eef_pos            => End Effector = 로봇 손 끝 위치
        robot0_eef_quat           => End Effector = 로봇 손 끝 쿼터니언 (4 dimensions)

        robot0_gripper_qpos       => 로봇 그리퍼 상태 (얼마나 열려있는지)
        robot0_gripper_qvel       => 로봇 그리퍼 속도

        cube_pos                  => 큐브 위치
        cube_quat                 => 큐브 쿼터니언

        gripper_to_cube_pos       => 그리퍼와 큐브 사이의 거리

        robot0_proprio-state      => 로봇 관절 위치, 속도, 그리퍼 위치, 속도
        object-state              => 큐브 위치, 쿼터니언


        입력은 obs이므로

        obs = [
            eef_position,
            eef_rotation,
            gripper_state,
            cube_position,
            cube_rotation,
        ]

        EEF xyz       3
        EEF quat      4
        gripper       ?
        cube xyz      3
        cube quat     4
        ...
        ────────────────
        observation vector

        이런 형태로 만들어진다.
"""

"""
        action
        │
        ├─ Δx
        ├─ Δy
        ├─ Δz
        │
        ├─ Δrotation x
        ├─ Δrotation y
        ├─ Δrotation z
        │
        └─ gripper

        7D action구조로 이루어져 있다.
        하지만 실제 action은 어떤 controller를 쓰느냐에 따라 달라진다.
"""
