"""Feature extractors and SB3 policy kwargs for the two compared architectures."""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from martha_nav.sim2d.observation import N_SECTORS


class LidarCnnExtractor(BaseFeaturesExtractor):
    """Circular 1D CNN over the LiDAR sectors plus a linear branch for the rest."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=160)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2, padding_mode='circular'), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2, padding_mode='circular'), nn.ReLU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, 1, N_SECTORS)).shape[1]
        self.lidar_head = nn.Sequential(nn.Linear(n_flat, 128), nn.ReLU())
        rest = observation_space.shape[0] - N_SECTORS
        self.rest_head = nn.Sequential(nn.Linear(rest, 32), nn.ReLU())

    def forward(self, obs):
        lidar = self.cnn(obs[:, None, :N_SECTORS])
        return torch.cat([self.lidar_head(lidar), self.rest_head(obs[:, N_SECTORS:])], dim=1)


def policy_kwargs(arch):
    """SB3 policy_kwargs for 'cnn' (main) or 'mlp' (baseline)."""
    kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                  share_features_extractor=False, log_std_init=-0.5)
    if arch == 'cnn':
        kwargs['features_extractor_class'] = LidarCnnExtractor
    elif arch != 'mlp':
        raise ValueError(f'unknown arch {arch!r}')
    return kwargs
