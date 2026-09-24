from dataclasses import replace

import numpy as np
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


def test_stall_penalty_turns_stalls_into_terminal_episodes():
    from martha_nav.sim2d.reward import RewardConfig
    env = NavEnv(replace(OPEN, reward=RewardConfig(stalled=-5.0)))
    env.reset(seed=3)
    while True:
        _, r, term, trunc, info = env.step(np.zeros(2))
        if term or trunc:
            break
    assert term and not trunc and info['outcome'] == 'stalled'
    assert info['r_stalled'] == -5.0


def test_lidar_encoding_reaches_the_observation():
    a, _ = NavEnv(OPEN).reset(seed=4)
    b, _ = NavEnv(replace(OPEN, lidar_encoding='linear')).reset(seed=4)
    assert not np.allclose(a[:90], b[:90]) and np.allclose(a[90:], b[90:])


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


def test_stuck_signal_grows_while_the_robot_does_not_advance():
    from martha_nav.sim2d.observation import obs_dim
    cfg = EnvConfig(stuck_signal=True, episode_seeds=(1000007,),
                    scenario=ScenarioConfig(sources=('lab',), obstacle_mode='always'))
    env = NavEnv(cfg)
    obs, _ = env.reset()
    assert obs.shape == (obs_dim(2, stuck_signal=True),)
    assert obs[-1] == 0.0
    # Spinning in place makes no progress along the route, so the signal must rise.
    signals = []
    for _ in range(10):
        obs, _, term, trunc, _ = env.step(np.array([0.0, 1.0]))
        signals.append(float(obs[-1]))
        if term or trunc:
            break
    assert signals == sorted(signals) and signals[-1] > 0.0


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
