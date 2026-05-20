from __future__ import annotations

import pytest

from lerobot_teleoperator_gamepad.record_manual import (
    ManualGamepadRecorder,
    is_record_toggle_action,
    run_manual_recording_session,
)


class IdentityProcessor:
    def __init__(self):
        self.reset_calls = 0

    def __call__(self, value):
        return value

    def reset(self):
        self.reset_calls += 1


class TeleopActionProcessor(IdentityProcessor):
    def __call__(self, value):
        action, _obs = value
        return action


class RobotActionProcessor(IdentityProcessor):
    def __call__(self, value):
        action, _obs = value
        return action


class FakeTeleop:
    def __init__(self, actions):
        self.actions = list(actions)

    def get_action(self):
        if self.actions:
            return self.actions.pop(0)
        return {"target_x": 0.0}


class FakeRobot:
    def __init__(self):
        self.sent_actions = []
        self.observation_count = 0

    def get_observation(self):
        self.observation_count += 1
        return {"ee_x": float(self.observation_count)}

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        return dict(action)


class FakeDataset:
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["ee_x"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["target_x"]},
    }
    num_episodes = 0

    def __init__(self):
        self.frames = []
        self.saved = 0
        self.cleared = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1
        self.num_episodes += 1
        self.frames.clear()

    def clear_episode_buffer(self):
        self.cleared += 1
        self.frames.clear()


def make_recorder(actions, *, max_episode_time_s=0.0):
    return ManualGamepadRecorder(
        robot=FakeRobot(),
        teleop=FakeTeleop(actions),
        dataset=FakeDataset(),
        fps=30,
        single_task="test task",
        teleop_action_processor=TeleopActionProcessor(),
        robot_action_processor=RobotActionProcessor(),
        robot_observation_processor=IdentityProcessor(),
        max_episode_time_s=max_episode_time_s,
        sleep_fn=lambda _seconds: None,
    )


def test_start_toggle_waits_without_recording_or_sending_actions():
    recorder = make_recorder([{"target_x": 0.1}, {"quit": True}])

    assert recorder.wait_for_start()

    assert recorder.dataset.frames == []
    assert recorder.robot.sent_actions == []


def test_record_episode_stops_on_second_start_without_recording_toggle_action():
    recorder = make_recorder([{"target_x": 0.1}, {"target_x": 0.2}, {"quit": True}])

    frames = recorder.record_episode()

    assert frames == 2
    assert recorder.robot.sent_actions == [{"target_x": 0.1}, {"target_x": 0.2}]
    assert [frame["action"].tolist() for frame in recorder.dataset.frames] == [pytest.approx([0.1]), pytest.approx([0.2])]


def test_session_saves_episode_and_returns_to_waiting_until_num_episodes():
    recorder = make_recorder(
        [
            {"quit": True},
            {"target_x": 0.1},
            {"quit": True},
            {"quit": True},
            {"target_x": 0.2},
            {"quit": True},
        ]
    )

    recorded = run_manual_recording_session(recorder=recorder, num_episodes=2, play_sounds=False)

    assert recorded == 2
    assert recorder.dataset.saved == 2
    assert recorder.robot.sent_actions == [{"target_x": 0.1}, {"target_x": 0.2}]


def test_zero_episode_time_does_not_timeout_before_stop_toggle():
    recorder = make_recorder([{"target_x": 0.1}, {"target_x": 0.2}, {"quit": True}], max_episode_time_s=0.0)

    frames = recorder.record_episode()

    assert frames == 2


def test_toggle_detection_uses_quit_control_bit():
    assert is_record_toggle_action({"quit": True})
    assert not is_record_toggle_action({"quit": False})
    assert not is_record_toggle_action({"target_x": 0.0})
