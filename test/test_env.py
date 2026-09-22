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
