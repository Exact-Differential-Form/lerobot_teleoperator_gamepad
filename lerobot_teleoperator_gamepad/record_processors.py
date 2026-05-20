from lerobot.processor import (
    RobotProcessorPipeline,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action

from .compat import RobotAction, RobotObservation
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
