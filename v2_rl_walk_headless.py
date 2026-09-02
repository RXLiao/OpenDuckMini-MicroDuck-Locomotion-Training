"""Real-robot deployment for a 101-input, 10-output leg-only PPO policy."""

import argparse
import os
import pickle
import time

import numpy as np
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.rl_utils import LowPassActionFilter, make_action_dict
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.xbox_controller import XBoxController


HOME_DIR = os.path.expanduser("~")
HEAD_ACTION_IDS = np.array([5, 6, 7, 8], dtype=np.int64)
LEG_ACTION_IDS = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], dtype=np.int64)


class RLWalk:
    def __init__(
        self,
        onnx_model_path,
        duck_config_path=f"{HOME_DIR}/duck_config.json",
        serial_port="/dev/ttyACM0",
        control_freq=50,
        pid=None,
        action_scale=0.25,
        commands=True,
        pitch_bias=0.0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
        command_ramp_duration=1.0,
    ):
        self.duck_config = DuckConfig(config_json_path=duck_config_path)
        self.commands_enabled = commands
        self.policy = OnnxInfer(onnx_model_path, awd=True)
        self.num_dofs = 14
        self.control_freq = float(control_freq)
        self.control_period = 1.0 / self.control_freq
        self.pid = pid or [30, 0, 0]
        self.action_scale = float(action_scale)
        self.save_obs = save_obs
        self.saved_obs = []
        self.replay_obs = None
        if replay_obs is not None:
            with open(replay_obs, "rb") as stream:
                self.replay_obs = pickle.load(stream)

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        self.hwi = HWI(self.duck_config, serial_port)
        self.start_hardware()
        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )
        self.feet_contacts = FeetContacts()

        self.init_pos = np.asarray(
            list(self.hwi.init_pos.values()), dtype=np.float32
        )
        self.motor_targets = self.init_pos.copy()
        self.prev_motor_targets = self.init_pos.copy()
        self.last_action = np.zeros(14, dtype=np.float32)
        self.last_last_action = np.zeros(14, dtype=np.float32)
        self.last_last_last_action = np.zeros(14, dtype=np.float32)

        self.target_commands = np.zeros(7, dtype=np.float32)
        self.last_commands = np.zeros(7, dtype=np.float32)
        self.command_ramp_steps = max(
            1, int(round(float(command_ramp_duration) * self.control_freq))
        )
        self.command_ramp_step = 0
        self.paused = self.duck_config.start_paused

        if self.commands_enabled:
            self.xbox_controller = XBoxController(20)

        self.PRM = PolyReferenceMotion("./polynomial_coefficients.pkl")
        self.imitation_i = 0.0
        self.imitation_phase = np.zeros(2, dtype=np.float32)
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        if self.duck_config.eyes:
            self.eyes = Eyes()
        if self.duck_config.projector:
            self.projector = Projector()
        if self.duck_config.speaker:
            self.sounds = Sounds(
                volume=1.0, sound_directory="../mini_bdx_runtime/assets/"
            )
        if self.duck_config.antennas:
            self.antennas = Antennas()

    def start_hardware(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14
        kps[5:9] = [8, 8, 8, 8]
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()
        time.sleep(2)

    def reset_command_ramp(self):
        self.command_ramp_step = 0
        self.last_commands[:] = 0.0

    def update_command_ramp(self):
        self.command_ramp_step = min(
            self.command_ramp_step + 1, self.command_ramp_steps
        )
        alpha = self.command_ramp_step / float(self.command_ramp_steps)
        self.last_commands = (alpha * self.target_commands).astype(np.float32)

    def get_obs(self):
        imu_data = self.imu.get_data()
        dof_pos = self.hwi.get_present_positions(
            ignore=["left_antenna", "right_antenna"]
        )
        dof_vel = self.hwi.get_present_velocities(
            ignore=["left_antenna", "right_antenna"]
        )
        if dof_pos is None or dof_vel is None:
            return None
        if len(dof_pos) != 14 or len(dof_vel) != 14:
            print("Invalid joint feedback length")
            return None

        obs = np.concatenate(
            [
                np.asarray(imu_data["gyro"], dtype=np.float32),
                np.asarray(imu_data["accelero"], dtype=np.float32),
                self.last_commands,
                np.asarray(dof_pos, dtype=np.float32) - self.init_pos,
                np.asarray(dof_vel, dtype=np.float32) * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                np.asarray(self.feet_contacts.get(), dtype=np.float32),
                self.imitation_phase,
            ]
        ).astype(np.float32)
        if obs.shape != (101,) or not np.all(np.isfinite(obs)):
            print("Invalid observation", obs.shape)
            return None
        return obs

    def expand_action(self, leg_action):
        leg_action = np.asarray(leg_action, dtype=np.float32).reshape(-1)
        if leg_action.shape != (10,):
            raise RuntimeError(
                f"leg-only ONNX must output 10 actions, got {leg_action.shape}"
            )
        full_action = np.zeros(14, dtype=np.float32)
        full_action[LEG_ACTION_IDS] = leg_action
        # Keep the externally controlled head in action history too, matching
        # the training environment's full 14-joint observation history.
        full_action[HEAD_ACTION_IDS] = (
            self.last_commands[3:7] / self.action_scale
        )
        return full_action

    def read_controller(self):
        raw, self.buttons, left_trigger, right_trigger = (
            self.xbox_controller.get_last_command()
        )
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.shape == (7,) and np.all(np.isfinite(raw)):
            self.target_commands = raw.copy()

        if self.buttons.dpad_up.triggered:
            self.phase_frequency_factor_offset += 0.05
        if self.buttons.dpad_down.triggered:
            self.phase_frequency_factor_offset -= 0.05
        self.phase_frequency_factor = 1.3 if self.buttons.LB.is_pressed else 1.0
        if self.buttons.X.triggered and self.duck_config.projector:
            self.projector.switch()
        if self.buttons.B.triggered and self.duck_config.speaker:
            self.sounds.play_random_sound()
        if self.duck_config.antennas:
            self.antennas.set_position_left(right_trigger)
            self.antennas.set_position_right(left_trigger)
        if self.buttons.A.triggered:
            self.paused = not self.paused
            if self.paused:
                print("PAUSE")
            else:
                self.reset_command_ramp()
                print("UNPAUSE: command ramp reset")

    def run(self):
        index = 0
        filter_start = time.time()
        self.reset_command_ramp()
        try:
            while True:
                loop_start = time.time()
                if self.commands_enabled:
                    self.read_controller()
                if self.paused:
                    time.sleep(0.1)
                    continue

                self.update_command_ramp()
                self.imitation_i = (
                    self.imitation_i
                    + self.phase_frequency_factor
                    + self.phase_frequency_factor_offset
                ) % self.PRM.nb_steps_in_period
                phase = self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                self.imitation_phase = np.array(
                    [np.cos(phase), np.sin(phase)], dtype=np.float32
                )
                obs = self.get_obs()
                if obs is None:
                    continue
                if self.save_obs:
                    self.saved_obs.append(obs.copy())
                if self.replay_obs is not None:
                    if index >= len(self.replay_obs):
                        break
                    obs = np.asarray(self.replay_obs[index], dtype=np.float32)

                full_action = self.expand_action(self.policy.infer(obs))
                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = full_action.copy()
                self.motor_targets = (
                    self.init_pos + full_action * self.action_scale
                )

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    if time.time() - filter_start > 1.0:
                        self.motor_targets = np.asarray(
                            self.action_filter.get_filtered_action(),
                            dtype=np.float32,
                        )

                self.prev_motor_targets = self.motor_targets.copy()
                # Do not add head commands again: expand_action already made
                # head targets equal init_pos[5:9] + last_commands[3:7].
                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )
                self.hwi.set_position_all(action_dict)

                index += 1
                elapsed = time.time() - loop_start
                if elapsed > self.control_period:
                    print("Policy control budget exceeded by", elapsed-self.control_period)
                time.sleep(max(0.0, self.control_period - elapsed))
        except KeyboardInterrupt:
            pass
        finally:
            if self.duck_config.antennas:
                self.antennas.stop()
            if self.duck_config.eyes:
                self.eyes.stop()
            if self.duck_config.projector:
                self.projector.stop()
            self.feet_contacts.stop()
            if self.save_obs:
                with open("robot_saved_obs.pkl", "wb") as stream:
                    pickle.dump(self.saved_obs, stream)
            print("TURNING OFF")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", required=True)
    parser.add_argument(
        "--duck_config_path", default=f"{HOME_DIR}/duck_config.json"
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0.0)
    parser.add_argument("--commands", action="store_true", default=True)
    parser.add_argument("--save_obs", action="store_true")
    parser.add_argument("--replay_obs", default=None)
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument("--command_ramp_duration", type=float, default=1.0)
    args = parser.parse_args()
    RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=[args.p, args.i, args.d],
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
        command_ramp_duration=args.command_ramp_duration,
    ).run()


if __name__ == "__main__":
    main()
