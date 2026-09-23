"""Copy a .world and set its physics step and real-time target.

Ported from martha/martha/simulation_speed.py: Gazebo runs faster than real time
when real_time_update_rate is raised above 1 / max_step_size.
"""
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_SPEED_FACTOR = 20.0
MAX_STEP_SIZE = 0.01


def ensure_state_plugin(world, update_rate=50.0):
    """Add gazebo_ros_state to the world so /gazebo/model_states actually publishes.

    Loading it with "gzserver -s" gives it no <update_rate>, and then it stays quiet.
    """
    for plugin in world.findall('plugin'):
        if plugin.get('filename') == 'libgazebo_ros_state.so':
            return
    plugin = ET.SubElement(world, 'plugin')
    plugin.set('name', 'gazebo_ros_state')
    plugin.set('filename', 'libgazebo_ros_state.so')
    ros = ET.SubElement(plugin, 'ros')
    ET.SubElement(ros, 'namespace').text = '/gazebo'
    ET.SubElement(plugin, 'update_rate').text = repr(float(update_rate))


def create_scaled_world(source, speed_factor=1.0, physics_step_size=None, directory=None):
    """Write a copy of `source` with the requested speed; returns its path."""
    factor = float(speed_factor)
    if not 0.0 < factor <= MAX_SPEED_FACTOR:
        raise ValueError(f'speed_factor must be in (0, {MAX_SPEED_FACTOR}]')
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f'world does not exist: {source}')

    tree = ET.parse(source)
    world = tree.getroot().find('world')
    physics = world.find('physics')
    if physics is None:
        physics = ET.SubElement(world, 'physics')
        physics.set('type', 'ode')

    step = physics.find('max_step_size')
    if step is None:
        step = ET.SubElement(physics, 'max_step_size')
    if physics_step_size is not None:
        size = float(physics_step_size)
        if not 0.0 < size <= MAX_STEP_SIZE:
            raise ValueError(f'physics_step_size must be in (0, {MAX_STEP_SIZE}]')
        step.text = repr(size)
    step_size = float(step.text)

    rate = physics.find('real_time_update_rate')
    if rate is None:
        rate = ET.SubElement(physics, 'real_time_update_rate')
    rate.text = repr(factor / step_size)
    factor_node = physics.find('real_time_factor')
    if factor_node is not None:
        factor_node.text = repr(factor)

    ensure_state_plugin(world)

    directory = Path(directory) if directory else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkstemp(prefix=f'{source.stem}_', suffix='.world', dir=directory)[1])
    tree.write(out, encoding='utf-8', xml_declaration=True)
    return out
