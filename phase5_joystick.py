# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0

"""Joystick task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

try:
    from mujoco_playground._src.collision import geoms_colliding
except ModuleNotFoundError:
    import jax.numpy as jp

    def geoms_colliding(data, geom1, geom2):
        geom_pairs = data.contact.geom
        same_order = (geom_pairs[:, 0] == geom1) & (geom_pairs[:, 1] == geom2)
        reverse_order = (geom_pairs[:, 0] == geom2) & (geom_pairs[:, 1] == geom1)
        return jp.any(same_order | reverse_order)


from . import constants
from . import base as open_duck_mini_v2_base

from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    cost_stand_still,
    cost_head_pos,
    reward_alive,
)
from playground.open_duck_mini_v2.custom_rewards import reward_imitation


USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,
        noise_config=config_dict.create(
            level=1.0,
            action_min_delay=0,
            action_max_delay=3,
            imu_min_delay=0,
            imu_max_delay=3,
            scales=config_dict.create(
                hip_pos=0.03,
                knee_pos=0.05,
                ankle_pos=0.08,
                head_pos=0.02,
                joint_vel=2.5,
                gravity=0.1,
                linvel=0.1,
                gyro=0.1,
                accelerometer=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=6.0,
                tracking_ang_vel=2.5,
                torques=-1.0e-3,
                # Fine-tuning target: reduce the high-frequency joint target
                # changes seen in the July policy without suppressing gait.
                action_rate=-0.75,
                stand_still=-1.0,
                head_pos=-4.0,
                # Survival-only continuation: make every upright control step
                # more valuable without altering command tracking targets.
                alive=8.0,
                # Continuous margin to the actual fall condition.  Unlike a
                # terminal-only signal this teaches recovery before gravity-z
                # crosses zero.
                upright=0.0,
                # Active only for one second after a push.  This targets the
                # observed recovery failures without changing normal gait.
                push_recovery=2.0,
                imitation=1.5,
                # Symmetry fine-tuning did not improve the matched survival
                # evaluation, so it is deliberately disabled in phase 6.
                gait_symmetry=0.0,
            ),
            tracking_sigma=0.01,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        lin_vel_x=[-0.15, 0.15],
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-1.0, 1.0],
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-1.5, 1.5],
        head_roll_range=[-0.5, 0.5],
        head_range_factor=1.0,
    )


class Joystick(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Track a joystick command."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe("home").ctrl

        if USE_IMITATION_REWARD:
            self.PRM = PolyReferenceMotion(
                "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
            )

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,
                0.01,
                0.01,
                0.01,
                0.01,
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,
            ]
        )

        self._njoints = self._mj_model.njnt
        self._actuators = self._mj_model.nu

        ###add###
        print("nu:", self._mj_model.nu)
        print("actuator_names:", self.actuator_names)
        ###add###
        
        self._head_actuator_names = [
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
        ]
        self._head_ids_np = np.array(
            [self.actuator_names.index(name) for name in self._head_actuator_names],
            dtype=np.int32,
        )
        self._head_ids = jp.array(self._head_ids_np, dtype=jp.int32)

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        qpos_noise_scale = np.zeros(self._actuators)

        hip_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_hip" in j
        ]
        knee_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_knee" in j
        ]
        ankle_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_ankle" in j
        ]

        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
        qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
        qpos_noise_scale[self._head_ids_np] = self._config.noise_config.scales.head_pos

        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        print(f"head actuator names: {self._head_actuator_names}")
        print(f"head actuator ids: {self._head_ids_np}")

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )

        base_qpos = base_qpos.at[3:7].set(new_quat)
        qpos = self.set_floating_base_qpos(base_qpos, qpos)

        rng, key = jax.random.split(rng)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key,
            (self._actuators,),
            minval=0.5,
            maxval=1.5,
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)

        rng, key = jax.random.split(rng)
        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05),
            qvel,
        )

        ctrl = self.get_actuator_joints_qpos(qpos)
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        if USE_IMITATION_REWARD:
            current_reference_motion = self.PRM.get_reference_motion(
                cmd[0],
                cmd[1],
                cmd[2],
                0,
            )
        else:
            current_reference_motion = jp.zeros(0)

        info = {
            "rng": rng,
            "step": 0,
            "command": jp.zeros(7),
            "desired_command": cmd,
            "command_ramp_step": jp.array(0, dtype=jp.int32),
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "completed_swing_time": jp.zeros(2),
            "completed_swing_peak": jp.zeros(2),
            "completed_swing_valid": jp.zeros(2, dtype=bool),
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "steps_since_push": jp.array(1000, dtype=jp.int32),
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            "imitation_phase": jp.zeros(2),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["command_ramp_step"] = jp.minimum(
            state.info["command_ramp_step"] + 1, 50
        )
        ramp = state.info["command_ramp_step"] / 50.0
        state.info["command"] = ramp * state.info["desired_command"]
        if USE_IMITATION_REWARD:
            state.info["imitation_i"] += 1
            state.info["imitation_i"] = (
                state.info["imitation_i"] % self.PRM.nb_steps_in_period
            )
            state.info["imitation_phase"] = jp.array(
                [
                    jp.cos(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                    jp.sin(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                ]
            )
        else:
            state.info["imitation_i"] = 0

        if USE_IMITATION_REWARD:
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(
                state.info["command"][0],
                state.info["command"][1],
                state.info["command"][2],
                state.info["imitation_i"],
            )
        else:
            state.info["current_reference_motion"] = jp.zeros(0)

        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
            state.info["rng"],
            4,
        )

        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history

        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx[0]]

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable
        pushed = jp.linalg.norm(push) > 0.0
        state.info["steps_since_push"] = jp.where(
            pushed,
            0,
            jp.minimum(state.info["steps_since_push"] + 1, 1000),
        )

        qvel = state.data.qvel
        qvel = qvel.at[
            self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2
        ].set(
            push * push_magnitude
            + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )

        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        motor_targets = (
            self._default_actuator + action_w_delay * self._config.action_scale
        )

        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]

            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets - self._config.max_motor_velocity * self.dt,
                prev_motor_targets + self._config.max_motor_velocity * self.dt,
            )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt

        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        # Compare complete left/right swing cycles, not instantaneous mirrored
        # actions.  A healthy biped gait is anti-phase, so forcing the two legs
        # to have equal actions at the same instant would suppress walking.
        state.info["completed_swing_time"] = jp.where(
            first_contact,
            state.info["feet_air_time"],
            state.info["completed_swing_time"],
        )
        state.info["completed_swing_peak"] = jp.where(
            first_contact,
            state.info["swing_peak"],
            state.info["completed_swing_peak"],
        )
        state.info["completed_swing_valid"] = (
            state.info["completed_swing_valid"] | first_contact
        )

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
            first_contact,
            contact,
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        command_change = state.info["step"] > 500
        state.info["desired_command"] = jp.where(
            command_change,
            self.sample_command(cmd_rng),
            state.info["desired_command"],
        )
        state.info["command_ramp_step"] = jp.where(
            command_change, 0, state.info["command_ramp_step"]
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )

        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v

        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        contact: jax.Array,
    ) -> mjx_env.Observation:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)
        accelerometer = accelerometer.at[0].set(accelerometer[0] + 1.3)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history

        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)

        for i in self.backlash_idx_to_add:
            joint_backlash = jp.insert(joint_backlash, i, 0)

        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)

        state = jp.hstack(
            [
                noisy_gyro,
                noisy_accelerometer,
                info["command"],
                noisy_joint_angles - self._default_actuator,
                noisy_joint_vel * self._config.dof_vel_scale,
                info["last_act"],
                info["last_last_act"],
                info["last_last_last_act"],
                info["motor_targets"],
                contact,
                info["imitation_phase"],
            ]
        )

        accelerometer = self.get_accelerometer(data)
        accelerometer = accelerometer.at[0].set(accelerometer[0] + 1.3)

        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,
                accelerometer,
                gravity,
                linvel,
                global_angvel,
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,
                data.actuator_force,
                contact,
                feet_vel,
                info["feet_air_time"],
                info["current_reference_motion"],
                info["imitation_i"],
                info["imitation_phase"],
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics
        del done
        del first_contact

        joints_qpos = self.get_actuator_joints_qpos(data.qpos)
        joints_qvel = self.get_actuator_joints_qvel(data.qvel)

        ret = {
            "tracking_lin_vel": reward_tracking_lin_vel(
                info["command"],
                self.get_local_linvel(data),
                self._config.reward_config.tracking_sigma,
            ),
            "tracking_ang_vel": reward_tracking_ang_vel(
                info["command"],
                self.get_gyro(data),
                self._config.reward_config.tracking_sigma,
            ),
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "alive": reward_alive(),
            "upright": jp.clip(self.get_gravity(data)[-1], 0.0, 1.0),
            "push_recovery": self._reward_push_recovery(data, info),
            "gait_symmetry": self._cost_gait_symmetry(info),
            "imitation": reward_imitation(
                self.get_floating_base_qpos(data.qpos),
                self.get_floating_base_qvel(data.qvel),
                joints_qpos,
                joints_qvel,
                contact,
                info["current_reference_motion"],
                info["command"],
                USE_IMITATION_REWARD,
            ),
            "head_pos": cost_head_pos(
                joints_qpos,
                joints_qvel,
                info["command"],
                self._head_ids,
            ),
            "stand_still": cost_stand_still(
                info["command"],
                joints_qpos,
                joints_qvel,
                self._default_actuator,
                ignore_head=True,
            ),
        }

        return ret

    def _cost_gait_symmetry(self, info: dict[str, Any]) -> jax.Array:
        """Penalize unequal completed left/right swing cycles while moving."""
        both_valid = jp.all(info["completed_swing_valid"])
        moving = jp.linalg.norm(info["command"][:3]) > 0.01

        # Normalize by tolerances so duration and clearance contribute on a
        # comparable scale.  Clipping prevents rare failed cycles from
        # overwhelming the survival objective.
        duration_delta = jp.clip(
            (info["completed_swing_time"][0] - info["completed_swing_time"][1])
            / 0.15,
            -2.0,
            2.0,
        )
        peak_delta = jp.clip(
            (info["completed_swing_peak"][0] - info["completed_swing_peak"][1])
            / 0.02,
            -2.0,
            2.0,
        )
        cost = jp.square(duration_delta) + jp.square(peak_delta)
        return jp.nan_to_num(cost) * both_valid * moving

    def _reward_push_recovery(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Reward regaining an upright, low-angular-rate state after pushes."""
        active = info["steps_since_push"] < 50
        up = jp.clip(self.get_gravity(data)[-1], 0.0, 1.0)
        gyro_xy = self.get_gyro(data)[:2]
        calm = jp.exp(-0.25 * jp.sum(jp.square(gyro_xy)))
        return active * (0.7 * up + 0.3 * calm)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8, rng9 = jax.random.split(rng, 9)

        lin_vel_x = jax.random.uniform(
            rng1,
            minval=self._config.lin_vel_x[0],
            maxval=self._config.lin_vel_x[1],
        )
        lin_vel_y = jax.random.uniform(
            rng2,
            minval=self._config.lin_vel_y[0],
            maxval=self._config.lin_vel_y[1],
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        # Half of commands deliberately combine high lateral velocity and
        # high yaw rate, the dominant phase-6 early-failure condition.
        # Oversample difficult lateral+yaw combinations during training.
        hard_start = jax.random.bernoulli(rng9, p=0.25)
        hard_y = jp.sign(lin_vel_y) * (0.14 + 0.3 * jp.abs(lin_vel_y))
        hard_yaw = jp.sign(ang_vel_yaw) * (0.7 + 0.3 * jp.abs(ang_vel_yaw))
        lin_vel_y = jp.where(hard_start, hard_y, lin_vel_y)
        ang_vel_yaw = jp.where(hard_start, hard_yaw, ang_vel_yaw)

        neck_pitch = jax.random.uniform(
            rng5,
            minval=self._config.neck_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.neck_pitch_range[1] * self._config.head_range_factor,
        )
        head_pitch = jax.random.uniform(
            rng6,
            minval=self._config.head_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.head_pitch_range[1] * self._config.head_range_factor,
        )
        head_yaw = jax.random.uniform(
            rng7,
            minval=self._config.head_yaw_range[0] * self._config.head_range_factor,
            maxval=self._config.head_yaw_range[1] * self._config.head_range_factor,
        )
        head_roll = jax.random.uniform(
            rng8,
            minval=self._config.head_roll_range[0] * self._config.head_range_factor,
            maxval=self._config.head_roll_range[1] * self._config.head_range_factor,
        )

        return jp.where(
            jax.random.bernoulli(rng4, p=0.05),
            jp.zeros(7),
            jp.hstack(
                [
                    lin_vel_x,
                    lin_vel_y,
                    ang_vel_yaw,
                    neck_pitch,
                    head_pitch,
                    head_yaw,
                    head_roll,
                ]
            ),
        )
