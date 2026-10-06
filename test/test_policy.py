import numpy as np
import torch
from gymnasium import spaces

from martha_nav.learning.policy import LidarCnnExtractor, policy_kwargs
from martha_nav.sim2d.observation import OBS_DIM

OBS_SPACE = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)


def test_cnn_extractor_shape():
    ext = LidarCnnExtractor(OBS_SPACE)
    out = ext(torch.zeros(4, OBS_DIM))
    assert out.shape == (4, 160) and ext.features_dim == 160


def test_cnn_is_circular_over_the_lidar():
    """Rotating the scan by one full turn of sectors must not change the output;
    and a hit next to sector 0 must affect it like a hit next to sector 89."""
    ext = LidarCnnExtractor(OBS_SPACE)
    obs = torch.rand(1, OBS_DIM)
    rolled = obs.clone()
    rolled[0, :90] = torch.roll(obs[0, :90], 90)
    assert torch.allclose(ext(obs), ext(rolled))
    conv = ext.cnn[0]
    x = torch.zeros(1, 1, 90)
    x[0, 0, 89] = 1.0
    assert conv(x)[0, :, 0].abs().sum() > 0      # sector 89 leaks into sector 0


def test_policy_kwargs_use_the_cnn():
    kwargs = policy_kwargs()
    assert kwargs['features_extractor_class'] is LidarCnnExtractor
    assert kwargs['share_features_extractor'] is False
