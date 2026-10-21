import pytest

from martha_nav.ros.xbox_teleop import Presses, SpeedLevel, stick_command

CFG = {'axis_x': 1, 'axis_y': 0, 'axis_yaw': 2}


def test_sticks_map_to_forward_sideways_and_turn():
    axes = [0.5, 1.0, -1.0, 0.0, 0.0, 0.0]  # left stick up and half left, right stick right
    assert stick_command(axes, CFG, [0.2, 0.15, 0.5]) == pytest.approx((0.2, 0.075, -0.5))


def test_a_missing_axis_reads_zero():
    assert stick_command([0.0, 1.0], CFG, [0.2, 0.15, 0.5]) == pytest.approx((0.2, 0.0, 0.0))


def test_a_press_counts_once_while_held():
    presses = Presses()
    assert [presses.update(b) for b in ([0, 0], [1, 0], [1, 0], [1, 1], [0, 0], [1, 0])] == \
        [set(), {0}, set(), {1}, set(), {0}]


def test_speed_steps_up_and_down_within_the_ceiling_and_the_floor():
    speed = SpeedLevel([0.2, 0.15, 0.5], [0.5, 0.5, 1.2], 1.25)
    speed.change(+1)
    assert speed.scales == pytest.approx([0.25, 0.1875, 0.625])
    speed.change(+10)                    # capped where the first axis meets its maximum
    assert speed.scales == pytest.approx([0.48, 0.36, 1.2])
    speed.change(-20)
    assert speed.scales == pytest.approx([0.05, 0.0375, 0.125])
