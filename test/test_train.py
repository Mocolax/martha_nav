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
