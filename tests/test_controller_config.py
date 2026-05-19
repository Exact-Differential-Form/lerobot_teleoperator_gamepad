from __future__ import annotations

import math

import pytest

from lerobot_teleoperator_gamepad.controller_config import (
    load_keybind_config,
    sensitivity_config,
    validate_sensitivity_bounds,
)
from lerobot_teleoperator_gamepad.gamepad import EvdevXboxController


def test_load_keybind_config_parses_yaml_subset(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        sensitivity:
          default_sensitivity: 3.0
          min_sensitivity: 0.5
          max_sensitivity: 9.0
        xbox_buttons:
          reset: BTN_SELECT
        xbox_axes:
          forward: ABS_Y:-
        """
    )

    config = load_keybind_config(config_path)

    assert config["xbox_buttons"]["reset"] == "BTN_SELECT"
    assert config["xbox_axes"]["forward"] == "ABS_Y:-"
    assert sensitivity_config(config) == {
        "default_sensitivity": 3.0,
        "min_sensitivity": 0.5,
        "max_sensitivity": 9.0,
    }


def test_invalid_sensitivity_bounds_raise():
    with pytest.raises(ValueError):
        validate_sensitivity_bounds(0.0, 1.0)
    with pytest.raises(ValueError):
        validate_sensitivity_bounds(2.0, 1.0)


def test_config_file_sensitivity_and_cli_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        sensitivity:
          default_sensitivity: 2.0
          min_sensitivity: 0.25
          max_sensitivity: 8.0
        xbox_buttons:
          quit: BTN_START
        xbox_axes:
          forward: ABS_Y:-
        """
    )

    class Device:
        name = "fake xbox"
        path = "/dev/input/event-test"
        fd = 0
        def grab(self):
            return None

    monkeypatch.setattr(
        "lerobot_teleoperator_gamepad.gamepad.find_xbox_controller",
        lambda device_path=None: Device(),
    )

    controller = EvdevXboxController(config_path=str(config_path), grab_device=False)
    assert controller.sensitivity == 2.0
    assert controller.min_sensitivity == 0.25
    assert controller.max_sensitivity == 8.0
    assert controller.axis_binding_config == {"forward": "ABS_Y:-"}

    controller = EvdevXboxController(
        config_path=str(config_path),
        sensitivity=4.0,
        min_sensitivity=1.0,
        max_sensitivity=5.0,
        grab_device=False,
    )
    assert controller.sensitivity == 4.0
    assert controller.min_sensitivity == 1.0
    assert controller.max_sensitivity == 5.0


def test_first_active_axis_does_not_accumulate_stale_dt(monkeypatch):
    from lerobot_teleoperator_gamepad import gamepad as gamepad_module

    class AbsInfo:
        min = -32768
        max = 32767

    class Event:
        type = gamepad_module.ecodes.EV_ABS
        code = gamepad_module.ecodes.ecodes["ABS_Y"]
        value = -32768

    class Device:
        name = "fake xbox"
        path = "/dev/input/event-test"
        fd = 0
        def __init__(self):
            self.events = [[Event()], []]
        def read(self):
            return self.events.pop(0)
        def absinfo(self, code):
            return AbsInfo()

    device = Device()
    monkeypatch.setattr(gamepad_module.select, "select", lambda *args: ([device.fd], [], []) if device.events[0] else ([], [], []))
    times = iter([99.0, 100.0, 100.1])
    monkeypatch.setattr(gamepad_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(gamepad_module, "find_xbox_controller", lambda device_path=None: device)

    controller = EvdevXboxController(sensitivity=5.0, grab_device=False)
    controller.last_tick = 90.0

    assert controller.read_command(timeout=0.0) is None
    command = controller.read_command(timeout=0.0)

    assert command is not None
    assert command.action == "forward"
    assert math.isclose(command.pose_delta[0], 0.005, rel_tol=1e-6)
