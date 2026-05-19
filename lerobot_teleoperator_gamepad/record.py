import lerobot.scripts.lerobot_record as record_module
from lerobot.configs import parser
from lerobot.processor import (
    RobotProcessorPipeline,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.scripts.lerobot_record import RecordConfig

from .compat import RobotAction, RobotObservation, register_third_party_devices
from .processors import TrossenGamepadActionProcessorStep


def make_trossen_gamepad_teleop_action_processor(
    robot_config=None,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[TrossenGamepadActionProcessorStep.from_robot_config(robot_config)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
        name="trossen_gamepad_teleop_action_processor",
    )


def make_trossen_gamepad_processors(robot_config=None):
    return (
        make_trossen_gamepad_teleop_action_processor(robot_config),
        make_default_robot_action_processor(),
        make_default_robot_observation_processor(),
    )


@parser.wrap()
def record_trossen_gamepad(cfg: RecordConfig):
    original_make_default_processors = record_module.make_default_processors
    record_module.make_default_processors = lambda: make_trossen_gamepad_processors(cfg.robot)
    try:
        return record_module.record(cfg)
    finally:
        record_module.make_default_processors = original_make_default_processors


def main():
    register_third_party_devices()
    record_trossen_gamepad()


if __name__ == "__main__":
    main()
