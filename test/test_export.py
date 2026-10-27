"""The exported numpy policy must act exactly as the PyTorch one, without importing it."""
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from martha_nav.robots import ROBOTS
from martha_nav.ros.ppo_local_planner import load_policy
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


def save_fresh_martha_policy(folder):
    """A policy nobody trained, with the config.yaml of a Martha run; returns its model path."""
    from stable_baselines3 import PPO

    from martha_nav.learning.policy import policy_kwargs
    PPO('MlpPolicy', NavEnv(EnvConfig(robot='martha')), policy_kwargs=policy_kwargs(), seed=0,
        device='cpu').save(folder / 'best_model.zip')
    (folder / 'config.yaml').write_text(yaml.safe_dump({'env': {'robot': 'martha'}}))
    return folder / 'best_model.zip'


@pytest.fixture(scope='module')
def martha_npz(tmp_path_factory):
    from martha_nav.learning.export import export
    return export(save_fresh_martha_policy(tmp_path_factory.mktemp('martha')))


def rewritten(source, target, **settings):
    """A copy of an exported policy with these settings replaced (None removes one)."""
    data = dict(np.load(source))
    merged = {**json.loads(str(data['settings'])), **settings}
    np.savez(target, **{**data, 'settings': json.dumps({k: v for k, v in merged.items()
                                                        if v is not None})})
    return str(target)


def test_a_fresh_martha_policy_exports_exactly(tmp_path):
    policy = same_actions(save_fresh_martha_policy(tmp_path), NavEnv(EnvConfig(robot='martha')))
    assert policy.settings['robot'] == 'martha' and policy.settings['action_dim'] == 2


def test_the_export_carries_the_robot_profile(martha_npz):
    from martha_nav.ros.numpy_policy import NumpyPolicy
    recorded = json.loads(json.dumps(asdict(ROBOTS['martha'])))     # posts become a list
    assert NumpyPolicy(martha_npz).settings['robot_profile'] == recorded


def test_an_exported_policy_for_another_profile_of_its_robot_is_refused(martha_npz, tmp_path):
    profile = asdict(ROBOTS['martha'])
    same = rewritten(martha_npz, tmp_path / 'same.npz', robot_profile=profile)
    assert load_policy(same)[1]['robot'] == 'martha'
    unrecorded = rewritten(martha_npz, tmp_path / 'unrecorded.npz', robot_profile=None)
    assert load_policy(unrecorded)[1]['robot'] == 'martha'
    other = rewritten(martha_npz, tmp_path / 'other.npz',
                      robot_profile={**profile, 'lidar_min': 0.16})
    with pytest.raises(ValueError, match=r'different martha profile \(changed: lidar_min\)'):
        load_policy(other)


def test_a_zip_trained_for_another_profile_is_refused_before_it_is_loaded(tmp_path):
    other = {**asdict(ROBOTS['martha']), 'lidar_min': 0.16}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'robot': 'martha'},
                                                          'robot_profile': other}))
    with pytest.raises(ValueError, match=r'different martha profile \(changed: lidar_min\)'):
        load_policy(str(tmp_path / 'best_model.zip'))


@pytest.mark.skipif(not WIDE.exists(), reason='needs runs/wide_dyn_s0')
def test_the_final_model_exports_exactly(tmp_path):
    import shutil
    for name in ('best_model.zip', 'config.yaml'):
        shutil.copy(WIDE.with_name(name), tmp_path / name)
    from martha_nav.learning.evaluate import trained_env_config
    same_actions(tmp_path / 'best_model.zip', NavEnv(trained_env_config(WIDE)))


def test_the_robot_side_never_imports_pytorch():
    code = ('import sys\n'
            'from martha_nav.ros import ppo_local_planner, numpy_policy\n'
            "heavy = [m for m in ('torch', 'stable_baselines3', 'gymnasium') if m in sys.modules]\n"
            'assert not heavy, heavy\n')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_a_policy_of_another_architecture_fails_when_it_is_loaded(martha_npz, tmp_path):
    from martha_nav.ros.numpy_policy import NumpyPolicy
    data = dict(np.load(martha_npz))
    name = 'pi_features_extractor.lidar_head.0.weight'
    data[name] = data[name][:, :-1]
    np.savez(tmp_path / 'wrong.npz', **data)
    with pytest.raises(ValueError):
        NumpyPolicy(tmp_path / 'wrong.npz')


def test_export_returns_the_path_it_wrote(tmp_path):
    from martha_nav.learning.export import export
    out = export(save_fresh_martha_policy(tmp_path), tmp_path / 'martha')
    assert out == tmp_path / 'martha.npz' and out.exists()


def test_the_node_runs_an_exported_policy_without_pytorch(martha_npz):
    code = ('import sys\n'
            'import rclpy\n'
            'from martha_nav.ros.ppo_local_planner import PpoLocalPlanner\n'
            f"rclpy.init(args=['--ros-args', '-p', 'checkpoint:={martha_npz}'])\n"
            'node = PpoLocalPlanner()\n'
            "assert node.core.robot.name == 'martha'\n"
            "heavy = [m for m in ('torch', 'stable_baselines3', 'gymnasium') if m in sys.modules]\n"
            'assert not heavy, heavy\n')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
