"""The nodes a launch file builds, with their evaluated parameters, without starting anything."""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

LAUNCH_DIR = Path(__file__).resolve().parents[1] / 'launch'


def built(name, arguments, only=None):
    """{executable: its parameters} of <name>.launch.py with these launch arguments.

    only limits the nodes whose parameters are evaluated (the others may need xacro).
    """
    spec = importlib.util.spec_from_file_location(f'{name}_launch', LAUNCH_DIR / f'{name}.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update(arguments)
    nodes = [n for n in module.launch_setup(context) if isinstance(n, Node)]
    return {n.node_executable: {k: v for part in evaluate_parameters(context, n._Node__parameters)
                                if isinstance(part, dict) for k, v in part.items()}
            for n in nodes if only is None or n.node_executable in only}
