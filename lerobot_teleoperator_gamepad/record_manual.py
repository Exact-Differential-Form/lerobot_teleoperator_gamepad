import logging
import time
from dataclasses import asdict
from pprint import pformat
from typing import Callable

import lerobot.scripts.lerobot_record as record_module
from lerobot.configs import parser
from lerobot.processor import (
    RobotProcessorPipeline,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.utils.constants import ACTION, OBS_STR

from .compat import RobotAction, RobotObservation, register_third_party_devices
from .record_processors import make_trossen_gamepad_teleop_action_processor

SleepFn = Callable[[float], None]
ClockFn = Callable[[], float]


def is_record_toggle_action(action: RobotAction | None) -> bool:
    return bool(action and action.get("quit", False))


class ManualGamepadRecorder:
    def __init__(
        self,
        *,
        robot,
        teleop,
        dataset,
        fps: int,
        single_task: str,
        teleop_action_processor: RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ],
        robot_action_processor: RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ],
        robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
        max_episode_time_s: float = 0.0,
        display_data: bool = False,
        display_compressed_images: bool = False,
        poll_interval_s: float = 0.05,
        sleep_fn: SleepFn = record_module.precise_sleep,
        clock: ClockFn = time.perf_counter,
    ):
        self.robot = robot
        self.teleop = teleop
        self.dataset = dataset
        self.fps = int(fps)
        self.single_task = single_task
        self.teleop_action_processor = teleop_action_processor
        self.robot_action_processor = robot_action_processor
        self.robot_observation_processor = robot_observation_processor
        self.max_episode_time_s = float(max_episode_time_s)
        self.display_data = display_data
        self.display_compressed_images = display_compressed_images
        self.poll_interval_s = float(poll_interval_s)
        self.sleep_fn = sleep_fn
        self.clock = clock
        self.recording = False
        self.current_episode_frames = 0

    def wait_for_start(self, events: dict[str, bool] | None = None) -> bool:
        logging.info("Waiting for gamepad Start to begin recording.")
        while not _should_stop(events):
            action = self.teleop.get_action()
            if is_record_toggle_action(action):
                logging.info("Gamepad Start pressed; starting episode.")
                return True
            self.sleep_fn(self.poll_interval_s)
        return False

    def record_episode(self, events: dict[str, bool] | None = None) -> int:
        self._reset_episode_state()
        self.recording = True
        self.current_episode_frames = 0
        start_episode_t = self.clock()
        control_interval = 1.0 / self.fps

        logging.info("Recording manual episode. Press gamepad Start to stop and save.")
        try:
            while not _should_stop(events):
                start_loop_t = self.clock()
                if self.max_episode_time_s > 0.0 and start_loop_t - start_episode_t >= self.max_episode_time_s:
                    logging.info("Manual episode reached safety timeout %.3fs.", self.max_episode_time_s)
                    break

                raw_action = self.teleop.get_action()
                if is_record_toggle_action(raw_action):
                    logging.info("Gamepad Start pressed; stopping episode.")
                    break

                obs = self.robot.get_observation()
                obs_processed = self.robot_observation_processor(obs)
                observation_frame = record_module.build_dataset_frame(
                    self.dataset.features,
                    obs_processed,
                    prefix=OBS_STR,
                )

                act_processed_teleop = self.teleop_action_processor((raw_action, obs))
                robot_action_to_send = self.robot_action_processor((act_processed_teleop, obs))
                sent_action = self.robot.send_action(robot_action_to_send)
                action_values = sent_action if sent_action is not None else act_processed_teleop

                action_frame = record_module.build_dataset_frame(
                    self.dataset.features,
                    action_values,
                    prefix=ACTION,
                )
                frame = {**observation_frame, **action_frame, "task": self.single_task}
                self.dataset.add_frame(frame)
                self.current_episode_frames += 1

                if self.display_data:
                    record_module.log_rerun_data(
                        observation=obs_processed,
                        action=action_values,
                        compress_images=self.display_compressed_images,
                    )

                dt_s = self.clock() - start_loop_t
                sleep_time_s = control_interval - dt_s
                if sleep_time_s < 0.0:
                    logging.warning(
                        "Manual record loop is running slower (%.1f Hz) than the target FPS (%s Hz).",
                        1.0 / dt_s if dt_s > 0.0 else 0.0,
                        self.fps,
                    )
                self.sleep_fn(max(sleep_time_s, 0.0))
        finally:
            self.recording = False

        return self.current_episode_frames

    def _reset_episode_state(self) -> None:
        for processor in (
            self.teleop_action_processor,
            self.robot_action_processor,
            self.robot_observation_processor,
        ):
            reset = getattr(processor, "reset", None)
            if callable(reset):
                reset()


def run_manual_recording_session(
    *,
    recorder: ManualGamepadRecorder,
    num_episodes: int,
    events: dict[str, bool] | None = None,
    play_sounds: bool = True,
) -> int:
    recorded_episodes = 0
    try:
        while recorded_episodes < num_episodes and not _should_stop(events):
            record_module.log_say(
                f"Press gamepad Start to record episode {recorder.dataset.num_episodes}",
                play_sounds,
            )
            if not recorder.wait_for_start(events):
                break

            frames = recorder.record_episode(events)
            if frames > 0:
                recorder.dataset.save_episode()
                recorded_episodes += 1
                record_module.log_say("Saved manual episode", play_sounds)
            else:
                logging.info("Manual episode had no frames; clearing buffer.")
                recorder.dataset.clear_episode_buffer()
    except KeyboardInterrupt:
        if recorder.recording and recorder.current_episode_frames > 0:
            logging.info(
                "Interrupted during manual episode; saving %s buffered frames.",
                recorder.current_episode_frames,
            )
            recorder.dataset.save_episode()
            recorded_episodes += 1
        else:
            logging.info("Interrupted while waiting; no episode to save.")
            recorder.dataset.clear_episode_buffer()
    return recorded_episodes


def create_manual_dataset(cfg: RecordConfig, robot, dataset_features):
    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
    if cfg.resume:
        dataset = record_module.LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
            image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
            image_writer_threads=(
                cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0
            ),
        )
        record_module.sanity_check_dataset_robot_compatibility(
            dataset,
            robot,
            cfg.dataset.fps,
            dataset_features,
        )
        return dataset

    record_module.sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
    return record_module.LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        encoder_threads=cfg.dataset.encoder_threads,
    )


@parser.wrap()
def record_trossen_gamepad_manual(cfg: RecordConfig):
    record_module.init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.policy is not None:
        raise ValueError("Manual gamepad recording supports teleop control only, not policy control.")
    if cfg.teleop is None:
        raise ValueError("Manual gamepad recording requires --teleop.type=trossen_gamepad_cartesian_teleop.")

    if cfg.display_data:
        record_module.init_rerun(session_name="manual_recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = record_module.make_robot_from_config(cfg.robot)
    teleop = record_module.make_teleoperator_from_config(cfg.teleop)

    teleop_action_processor = make_trossen_gamepad_teleop_action_processor(cfg.robot)
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()

    dataset_features = record_module.combine_feature_dicts(
        record_module.aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=record_module.create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        record_module.aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=record_module.create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    try:
        dataset = create_manual_dataset(cfg, robot, dataset_features)
        robot.connect()
        teleop.connect()
        listener, events = record_module.init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. Consider --dataset.streaming_encoding=true "
                "--dataset.encoder_threads=2 for faster episode saving."
            )

        logging.info("Manual Start/Stop recorder ready. Press gamepad Start to begin.")
        record_module.log_say(
            "Manual Start/Stop recorder ready. Press gamepad Start to begin.",
            cfg.play_sounds,
        )

        recorder = ManualGamepadRecorder(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            fps=cfg.dataset.fps,
            single_task=cfg.dataset.single_task,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            max_episode_time_s=cfg.dataset.episode_time_s,
            display_data=cfg.display_data,
            display_compressed_images=display_compressed_images,
        )

        with record_module.VideoEncodingManager(dataset):
            run_manual_recording_session(
                recorder=recorder,
                num_episodes=cfg.dataset.num_episodes,
                events=events,
                play_sounds=cfg.play_sounds,
            )
    finally:
        record_module.log_say("Stop manual recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if getattr(robot, "is_connected", False):
            robot.disconnect()
        if getattr(teleop, "is_connected", False):
            teleop.disconnect()

        if not record_module.is_headless() and listener:
            listener.stop()

        if dataset and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        record_module.log_say("Exiting", cfg.play_sounds)
    return dataset


def _should_stop(events: dict[str, bool] | None) -> bool:
    return bool(events and (events.get("stop_recording") or events.get("exit_early")))


def main():
    register_third_party_devices()
    record_trossen_gamepad_manual()


if __name__ == "__main__":
    main()
