from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_package_imports():
    import martha_nav.learning  # noqa: F401
    import martha_nav.sim2d  # noqa: F401


def test_worlds_copied():
    names = sorted(p.stem for p in (REPO / 'worlds').glob('*.world'))
    assert names == ['four_rooms', 'hall', 'lab', 'multi', 'roblab', 'room', 'tube']


EXECUTABLES = ['esp32_bridge', 'evaluate_2d', 'evaluate_gazebo', 'export_policy',
               'gazebo_ground_truth_tf', 'global_planner', 'mecanum_cmd_vel_bridge',
               'ppo_local_planner', 'train_policy', 'world_map_publisher']


def entry_points():
    import re
    return dict(re.findall(r"'(\w+) = (martha_nav\.[\w.]+):main'", (REPO / 'setup.py').read_text()))


def test_executables_say_what_they_do():
    """ros2 run martha_nav <name>: the names are the package's interface."""
    import importlib
    points = entry_points()
    assert sorted(points) == EXECUTABLES
    for module in points.values():
        assert callable(importlib.import_module(module).main)


def test_the_launch_files_run_installed_executables():
    import re
    for launch in ('sim.launch.py', 'real.launch.py', 'burger.launch.py'):
        launched = re.findall(r"package='martha_nav', executable='(\w+)'",
                              (REPO / 'launch' / launch).read_text())
        assert launched and set(launched) <= set(entry_points()), launch
