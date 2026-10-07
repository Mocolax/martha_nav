"""The trained policy's deterministic action in numpy, so the robot needs no PyTorch.

export_policy (martha_nav/learning/export.py) writes the .npz read here: the weights of the
actor (CNN extractor, policy MLP, action layer), the settings the observation needs and the
profile of the robot it was trained for, which must still be the current one.
"""
import json

import numpy as np

from martha_nav.robots import check_profile
from martha_nav.sim2d.observation import N_SECTORS

SETTINGS = ('robot', 'action_dim', 'target')
CONVS = ((0, 1, 2), (2, 2, 2), (4, 2, 1))    # LidarCnnExtractor.cnn: (index, stride, padding)


def settings_of(cfg):
    """What the local planner needs from the EnvConfig a policy was trained with."""
    return {k: getattr(cfg, k) for k in SETTINGS}


def conv1d_circular(x, w, b, stride, pad):
    """torch's Conv1d(padding_mode='circular') on one sample: x (channels, length)."""
    xp = np.concatenate([x[:, -pad:], x, x[:, :pad]], axis=1)
    k = w.shape[2]
    n = (xp.shape[1] - k) // stride + 1
    windows = xp[:, np.arange(n)[:, None] * stride + np.arange(k)[None, :]]
    return np.einsum('ock,cnk->on', w, windows) + b[:, None]


class NumpyPolicy:
    """Same predict() as SB3's PPO, always deterministic, for one observation."""

    def __init__(self, path):
        data = np.load(path)
        self.settings = json.loads(str(data['settings']))
        check_profile(self.settings.get('robot_profile'), self.settings['robot'], path)
        self.w = {k: data[k].astype(np.float64) for k in data.files if k != 'settings'}

    def _linear(self, name, x):
        return self.w[f'{name}.weight'] @ x + self.w[f'{name}.bias']

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        x = obs[None, :N_SECTORS]
        for i, stride, pad in CONVS:
            name = f'pi_features_extractor.cnn.{i}'
            x = np.maximum(conv1d_circular(x, self.w[f'{name}.weight'], self.w[f'{name}.bias'],
                                           stride, pad), 0.0)
        lidar = np.maximum(self._linear('pi_features_extractor.lidar_head.0', x.reshape(-1)), 0.0)
        rest = np.maximum(self._linear('pi_features_extractor.rest_head.0', obs[N_SECTORS:]), 0.0)
        h = np.concatenate([lidar, rest])
        for i in (0, 2):                                   # Linear, Tanh, Linear, Tanh
            h = np.tanh(self._linear(f'mlp_extractor.policy_net.{i}', h))
        return np.clip(self._linear('action_net', h), -1.0, 1.0), None
