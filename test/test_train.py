import numpy as np
from stable_baselines3 import PPO

from martha_nav.learning.train import main as train_main
from martha_nav.sim2d.observation import OBS_DIM


def test_train_smoke(tmp_path):
    run_dir = train_main(['--preset', 'gate', '--arch', 'cnn', '--steps', '2048', '--n-envs', '2',
                          '--eval-every', '1024', '--eval-episodes', '2',
                          '--runs-dir', str(tmp_path), '--name', 'smoke'])
    for f in ('config.yaml', 'episodes.csv', 'evals.csv', 'best_model.zip', 'last_model.zip',
              'vecnormalize.pkl'):
        assert (run_dir / f).exists(), f
    model = PPO.load(run_dir / 'last_model.zip', device='cpu')
    action, _ = model.predict(np.zeros(OBS_DIM, np.float32), deterministic=True)
    assert action.shape == (2,)


def test_experiment_flags_reach_the_env_config():
    from martha_nav.learning.train import build_config
    cfg = build_config('full', collision=-20.0, stalled=-5.0, lidar_encoding='inverse')
    assert cfg.reward.collision == -20.0 and cfg.reward.stalled == -5.0
    assert cfg.lidar_encoding == 'inverse'
    default = build_config('full')
    assert default.reward.collision == -20.0 and default.reward.stalled == 0.0
    assert default.lidar_encoding == 'inverse'
    base = build_config('full', collision=-10.0, lidar_encoding='linear')   # reproduces full_cnn_s0
    assert base.reward.collision == -10.0 and base.lidar_encoding == 'linear'
