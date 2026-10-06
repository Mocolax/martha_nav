"""export_policy: a trained policy's actor and settings in one .npz, for the robot.

    ros2 run martha_nav export_policy --model runs/burger_s0/best_model.zip
-> runs/burger_s0/policy.npz, run with ppo_local_planner checkpoint:=.../policy.npz.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from martha_nav.learning import is_recurrent
from martha_nav.learning.evaluate import load_model, trained_env_config
from martha_nav.ros.numpy_policy import settings_of

ACTOR = ('pi_features_extractor.', 'mlp_extractor.policy_net.', 'action_net.')


def export(model_path, out=None):
    model = load_model(model_path)
    if is_recurrent(model):
        raise ValueError('only policies without an LSTM can be exported')
    weights = {k: v.cpu().numpy() for k, v in model.policy.state_dict().items()
               if k.startswith(ACTOR)}
    out = Path(out) if out else Path(model_path).with_name('policy.npz')
    np.savez(out, settings=json.dumps(settings_of(trained_env_config(model_path))), **weights)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Export a trained policy to numpy, for the robot.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default=None, help='default: policy.npz next to the model')
    args = ap.parse_args(argv)
    print(f'policy -> {export(args.model, args.out)}')
