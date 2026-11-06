import numpy as np

from martha_nav.learning.evaluate import load_model, run_episodes, summarize
from martha_nav.learning.pid import PidController
from martha_nav.robots import ROBOTS
from martha_nav.sim2d.env import EnvConfig, eval_seeds
from martha_nav.sim2d.observation import build_observation
from martha_nav.sim2d.scenarios import ScenarioConfig

ROBOT = ROBOTS['martha_tower']
ANGLES = np.linspace(-np.pi, np.pi, 360, endpoint=False)


def obs_with(ranges, carrot):
    return build_observation(ranges, ANGLES, (0.0, 0.0), carrot, (0.0, 0.0), robot=ROBOT)


def test_drives_straight_at_a_carrot_ahead_in_the_open():
    action, _ = PidController(ROBOT).predict(obs_with(np.full(360, 8.0), (2.0, 0.0)))
    assert action[0, 0] > 0.9 and abs(action[0, 1]) < 0.05


def test_stops_at_a_wall_and_turns_towards_the_freer_side():
    ranges = np.full(360, 8.0)
    ranges[np.abs(ANGLES) < np.radians(30)] = 0.25          # a wall 0.16 m from the bumper
    ranges[(ANGLES > np.radians(30)) & (ANGLES < np.radians(150))] = 0.6   # left is cramped
    action, _ = PidController(ROBOT).predict(obs_with(ranges, (2.0, 0.0)))
    assert action[0, 0] == 0 and action[0, 1] < 0                           # turns right


def test_a_run_marked_pid_loads_the_controller_and_reaches_goals(tmp_path):
    (tmp_path / 'config.yaml').write_text('controller: pid\nenv:\n  robot: martha_tower\n')
    model = load_model(tmp_path / 'best_model.zip')
    assert isinstance(model, PidController) and model.robot is ROBOT
    cfg = EnvConfig(robot='martha_tower',
                    scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none'))
    s = summarize(run_episodes(model, cfg, eval_seeds(4), n_envs=2))
    assert s['success'] == 1.0
