"""The exported numpy policy must act exactly as the PyTorch one, without importing it."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from martha_nav.sim2d.env import EnvConfig, NavEnv

WIDE = Path(__file__).resolve().parents[1] / 'runs' / 'wide_dyn_s0' / 'best_model.zip'


def same_actions(model_path, env, steps=40):
    from stable_baselines3 import PPO

    from martha_nav.learning.export import export
    from martha_nav.ros.numpy_policy import NumpyPolicy
    model = PPO.load(model_path, device='cpu')
    policy = NumpyPolicy(export(model_path))
    obs, _ = env.reset(seed=7)
    for _ in range(steps):
        expected, _ = model.predict(obs, deterministic=True)
        got, _ = policy.predict(obs)
        assert np.allclose(got, expected, atol=1e-5)
        obs, _, done, truncated, _ = env.step(expected)
        if done or truncated:
            obs, _ = env.reset()
    return policy


def test_a_fresh_burger_policy_exports_exactly(tmp_path):
    from stable_baselines3 import PPO

    from martha_nav.learning.policy import policy_kwargs
    env = NavEnv(EnvConfig(robot='burger'))
    PPO('MlpPolicy', env, policy_kwargs=policy_kwargs(), seed=0,
        device='cpu').save(tmp_path / 'best_model.zip')
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'robot': 'burger'}}))
    policy = same_actions(tmp_path / 'best_model.zip', env)
    assert policy.settings['robot'] == 'burger' and policy.settings['action_dim'] == 2


@pytest.mark.skipif(not WIDE.exists(), reason='needs runs/wide_dyn_s0')
def test_the_final_model_exports_exactly(tmp_path):
    import shutil
    for name in ('best_model.zip', 'config.yaml'):
        shutil.copy(WIDE.with_name(name), tmp_path / name)
    from martha_nav.learning.evaluate import trained_env_config
    same_actions(tmp_path / 'best_model.zip', NavEnv(trained_env_config(WIDE)))
