from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from martha_nav.sim2d.env import TRAIN_SEED_LIMIT, EnvConfig, NavEnv
from martha_nav.sim2d.scenarios import ScenarioConfig

OPEN = EnvConfig(scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none'))


def pursue(obs):
    """Scripted controller: turn toward the carrot, drive when roughly aligned."""
    ang = obs[91]
    return np.array([1.0 if abs(ang) < 0.25 else 0.0, np.clip(4 * ang, -1, 1)])


def test_gymnasium_api():
    check_env(NavEnv(), skip_render_check=True)


def test_reset_is_deterministic_per_seed():
    env = NavEnv()
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.array_equal(a, b)
    assert env.episode_seed < TRAIN_SEED_LIMIT


def test_episode_seeds_are_played_in_order():
    env = NavEnv(replace(OPEN, episode_seeds=(TRAIN_SEED_LIMIT + 5, TRAIN_SEED_LIMIT + 9)))
    env.reset()
    assert env.episode_seed == TRAIN_SEED_LIMIT + 5
    env.reset()
    assert env.episode_seed == TRAIN_SEED_LIMIT + 9


def test_scripted_controller_reaches_goals_in_open_rooms():
    env = NavEnv(OPEN)
    outcomes = []
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(pursue(obs))
            done = term or trunc
        outcomes.append(info['outcome'])
        if info['outcome'] == 'success':
            assert info['r_goal'] == 20.0 and 0.0 < info['spl'] <= 1.0
    assert outcomes.count('success') >= 18


def test_progress_is_paid_once_per_metre():
    env = NavEnv(OPEN)
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(pursue(obs))
        done = term or trunc
    assert info['outcome'] == 'success'
    assert info['r_progress'] <= info['route_length'] + 1e-6


def test_driving_into_a_wall_is_a_terminal_collision():
    env = NavEnv(OPEN)
    env.reset(seed=2)
    for _ in range(2000):
        _, r, term, trunc, info = env.step(np.array([1.0, 0.0]))
        if term or trunc:
            break
    # Straight ahead from a random pose ends at a wall or, rarely, at the goal.
    assert info['outcome'] in ('collision', 'success')
    if info['outcome'] == 'collision':
        assert r < -9.0


def test_standing_still_is_truncated_as_stalled():
    env = NavEnv(OPEN)
    env.reset(seed=3)
    steps = 0
    while True:
        _, _, term, trunc, info = env.step(np.zeros(2))
        steps += 1
        if term or trunc:
            break
    assert trunc and info['outcome'] == 'stalled' and steps == 150


def test_holonomic_env_matches_the_contract():
    from martha_nav.sim2d.observation import obs_dim
    env = NavEnv(replace(OPEN, action_dim=3))
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=5)
    assert obs.shape == (obs_dim(3),) and env.action_space.shape == (3,)
    obs, _, _, _, _ = env.step(np.array([0.5, 0.5, 0.0]))
    assert np.isfinite(obs).all()


def test_the_default_env_stays_two_dimensional():
    env = NavEnv()
    assert env.action_space.shape == (2,) and env.observation_space.shape == (96,)


def test_goal_target_points_the_observation_at_the_goal_not_the_carrot():
    from martha_nav.sim2d.observation import GOAL_MAX, N_SECTORS
    cfg = EnvConfig(target='goal', episode_seeds=(1000005,),
                    scenario=ScenarioConfig(sources=('lab',), obstacle_mode='always'))
    env = NavEnv(cfg)
    obs, _ = env.reset()
    x, y, th = env.dyn.pose
    dx, dy = env.sc.goal[0] - x, env.sc.goal[1] - y
    rel = (np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy)
    assert abs(obs[N_SECTORS] - min(np.hypot(dx, dy), GOAL_MAX) / GOAL_MAX) < 1e-5
    assert abs(obs[N_SECTORS + 1] - np.arctan2(rel[1], rel[0]) / np.pi) < 1e-5


def test_geodesic_progress_telescopes_to_the_distance_covered():
    from dataclasses import replace
    reward = replace(EnvConfig().reward, progress_mode='geodesic')
    env = NavEnv(EnvConfig(reward=reward, episode_seeds=(1000005,),
                           scenario=ScenarioConfig(sources=('lab',), obstacle_mode='always')))
    obs, _ = env.reset()
    start = env.geo
    for _ in range(40):
        obs, _, term, trunc, _ = env.step(pursue(obs))
        if term or trunc:
            break
    # Potential-based: the sum of the per-step terms is the net drop in distance.
    assert abs(env.terms['progress'] - (start - env.geo)) < 1e-6


def test_an_evaluation_episode_records_its_trajectory():
    """For the plots of the evasion: the pose every control step, from the start pose on."""
    from dataclasses import replace

    from martha_nav.sim2d.dynamics import DT
    from martha_nav.sim2d.env import EnvConfig, NavEnv, eval_seeds
    from martha_nav.sim2d.scenarios import ScenarioConfig
    cfg = EnvConfig(scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none'),
                    episode_seeds=tuple(eval_seeds(1)), record_trajectory=True)
    env = NavEnv(cfg)
    env.reset()
    start = env.sc.start
    info = {}
    while 'outcome' not in info:
        *_, info = env.step(np.array([0.5, 0.3]))
    traj = info['trajectory']
    assert len(traj) == info['steps'] + 1
    assert traj[0] == {'t': 0.0, 'x': start[0], 'y': start[1], 'yaw': start[2]}
    assert traj[-1]['t'] == pytest.approx(info['steps'] * DT)
    walked = sum(np.hypot(b['x'] - a['x'], b['y'] - a['y']) for a, b in zip(traj, traj[1:]))
    assert walked == pytest.approx(info['travelled'])
    env = NavEnv(replace(cfg, record_trajectory=False))            # training: nothing extra
    env.reset()
    info = {}
    while 'outcome' not in info:
        *_, info = env.step(np.array([0.5, 0.3]))
    assert 'trajectory' not in info


def test_the_burger_episode_uses_its_profile():
    from martha_nav.robots import ROBOTS
    from martha_nav.sim2d.env import episode_steps
    env = NavEnv(EnvConfig(robot='burger',
                           scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none')))
    env.reset(seed=3)
    assert env.robot is ROBOTS['burger']
    assert env.max_steps == episode_steps(env.sc.path.length, ROBOTS['burger'])
    assert episode_steps(10.0, ROBOTS['burger']) > episode_steps(10.0)      # slower robot, more time
    # Its LiDAR scans at 5 Hz: the scan changes every other control step.
    scans = []
    for _ in range(6):
        env.step(np.array([1.0, 0.0]))
        scans.append(env.ranges.copy())
    changed = [not np.array_equal(a, b) for a, b in zip(scans, scans[1:])]
    assert changed in ([True, False, True, False, True], [False, True, False, True, False])
    assert env.ranges.max() <= 3.5 and (env.ranges >= 0.12).all()


def test_a_holonomic_action_space_needs_a_holonomic_robot():
    with pytest.raises(ValueError, match='burger'):
        NavEnv(EnvConfig(robot='burger', action_dim=3))


def test_the_burger_lidar_reads_nothing_inside_its_blind_zone():
    from martha_nav.sim2d.geometry import draw_box
    env = NavEnv(EnvConfig(robot='burger', lidar_noise=(0.0, 0.0), lidar_dropout=0.0,
                           scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none')))
    env.reset(seed=3)
    x, y, th = env.dyn.pose
    ox, oy = x + env.robot.lidar_offset_x * np.cos(th), y + env.robot.lidar_offset_x * np.sin(th)
    # A wall from 0.05 m to 0.50 m in front of the LiDAR: its nearest cells are under 0.12 m away.
    draw_box(env.sc.full, ox + 0.275 * np.cos(th), oy + 0.275 * np.sin(th), 0.45, 2.0, th)
    env._scan()
    ahead = np.abs(env.ray_angles) < np.deg2rad(15)
    assert (env.ranges[ahead] == env.robot.lidar_range).all()
    assert (env.ranges >= env.robot.lidar_min).all()


def test_the_burger_collides_with_its_own_outline_not_marthas():
    from martha_nav.robots import ROBOTS
    from martha_nav.sim2d.geometry import draw_box, footprint_collides, footprint_points
    env = NavEnv(EnvConfig(robot='burger',
                           scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none')))
    env.reset(seed=3)
    assert np.array_equal(env.footprint, footprint_points(ROBOTS['burger']))
    x, y, th = env.dyn.pose
    # A wall 0.10 to 0.20 m ahead of base_link: Martha's rectangle reaches 0.28 m, the Burger's 0.04 m.
    draw_box(env.sc.full, x + 0.15 * np.cos(th), y + 0.15 * np.sin(th), 0.10, 1.0, th)
    assert footprint_collides(env.sc.full, x, y, th, footprint_points(ROBOTS['martha']))
    assert not footprint_collides(env.sc.full, x, y, th, env.footprint)
    for _ in range(30):
        *_, term, _, info = env.step(np.array([1.0, 0.0]))
        if term:
            break
    assert info['outcome'] == 'collision'


def test_the_tower_posts_appear_in_the_scan_wherever_the_robot_is():
    env = NavEnv(EnvConfig(robot='martha_tower', lidar_noise=(0.0, 0.0), lidar_dropout=0.0,
                           scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none')))
    env.reset(seed=3)
    env._scan()
    behind = np.abs(np.abs(np.rad2deg(env.ray_angles)) - 133) <= 2
    assert (env.ranges[behind] < 0.16).all()
    assert (env.ranges[np.abs(env.ray_angles) < np.deg2rad(90)] > 0.5).all()
