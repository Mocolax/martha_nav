import numpy as np
from stable_baselines3 import PPO

from martha_nav.learning.train import main as train_main
from martha_nav.sim2d.observation import OBS_DIM


def test_train_smoke(tmp_path):
    status = train_main(['--preset', 'gate', '--steps', '2048', '--n-envs', '2',
                         '--eval-every', '1024', '--eval-episodes', '2',
                         '--runs-dir', str(tmp_path), '--name', 'smoke'])
    assert status is None               # train_policy exits 0 (sys.exit(main()))
    run_dir = tmp_path / 'smoke'
    for f in ('config.yaml', 'episodes.csv', 'evals.csv', 'best_model.zip', 'last_model.zip',
              'vecnormalize.pkl'):
        assert (run_dir / f).exists(), f
    model = PPO.load(run_dir / 'last_model.zip', device='cpu')
    action, _ = model.predict(np.zeros(OBS_DIM, np.float32), deterministic=True)
    assert action.shape == (2,)


def test_experiment_flags_reach_the_env_config():
    from martha_nav.learning.train import build_config
    assert build_config('full').reward.collision == -20.0
    cfg = build_config('full', collision=-10.0, progress_mode='geodesic')
    assert cfg.reward.collision == -10.0 and cfg.reward.progress_mode == 'geodesic'


def test_build_config_takes_the_robot_and_its_inflation():
    from martha_nav.learning.train import build_config
    cfg = build_config('full', robot='burger')
    assert cfg.robot == 'burger' and cfg.scenario.inflation == 0.20
    assert build_config('full').scenario.inflation == 0.40


def test_ppo_updates_are_kl_limited():
    # Without it the policy std collapsed and approx_kl reached 1-2 (docs/resultados.md).
    from martha_nav.learning.train import PPO_PARAMS
    assert PPO_PARAMS['target_kl'] == 0.02


def test_recurrent_flag_selects_recurrent_ppo(tmp_path):
    """Arm L: an LSTM policy, trained with RecurrentPPO from sb3-contrib."""
    from sb3_contrib import RecurrentPPO

    from martha_nav.learning.train import build_model
    import martha_nav.learning.train as train_module
    env_cfg = train_module.build_config('gate')
    venv = train_module.make_vec_env(env_cfg, n_envs=2, seed=0)
    try:
        model = build_model(venv, recurrent=True, seed=0, device='cpu',
                            tensorboard_log=str(tmp_path))
        assert isinstance(model, RecurrentPPO)
        assert hasattr(model.policy, 'lstm_actor')          # an actual LSTM, not just the class
        assert model.policy.lstm_actor.hidden_size > 0
    finally:
        venv.close()


def test_wide_dynamics_flag_reaches_the_env_config():
    from martha_nav.learning.train import build_config
    from martha_nav.sim2d.dynamics import WIDE_DYNAMICS, DynamicsRanges
    assert build_config('full').dynamics == DynamicsRanges()
    assert build_config('full', wide_dynamics=True).dynamics == WIDE_DYNAMICS
