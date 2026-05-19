from __future__ import annotations

try:
    from lerobot.configs import PipelineFeatureType
except ImportError:  # LeRobot 0.4.0 exposes these from configs.types only.
    from lerobot.configs.types import PipelineFeatureType

try:
    from lerobot.types import RobotAction, RobotObservation, TransitionKey
except ModuleNotFoundError:  # LeRobot 0.4.0 keeps runtime aliases in processor.core.
    from lerobot.processor.core import RobotAction, RobotObservation, TransitionKey


def register_third_party_devices() -> None:
    try:
        from lerobot.utils.import_utils import register_third_party_plugins
    except ImportError:
        from lerobot.utils.import_utils import register_third_party_devices as register
    else:
        register = register_third_party_plugins
    register()
