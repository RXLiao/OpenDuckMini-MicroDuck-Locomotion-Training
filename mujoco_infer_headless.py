"""MuJoCo inference for a 10-output leg-only Open Duck PPO policy."""

import argparse

import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer


HEAD_ACTION_IDS = np.array([5, 6, 7, 8], dtype=np.int64)
LEG_ACTION_IDS = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], dtype=np.int64)


class LegPolicyAdapter:
    """Expand ten PPO leg actions to the fourteen simulated actuators."""

    def __init__(self, policy, owner):
        self.policy = policy
        self.owner = owner

    def infer(self, obs):
        leg_action = np.asarray(self.policy.infer(obs), dtype=np.float32).reshape(-1)
        if leg_action.shape != (10,):
            raise RuntimeError(
                f"leg-only ONNX must output (10,), received {leg_action.shape}"
            )
        full_action = np.zeros(14, dtype=np.float32)
        full_action[LEG_ACTION_IDS] = leg_action
        # MjInfer later computes default + full_action * action_scale.  This
        # gives exactly default_head + commands[3:7], independent of PPO.
        full_action[HEAD_ACTION_IDS] = (
            np.asarray(self.owner.commands[3:7], dtype=np.float32)
            / self.owner.action_scale
        )
        return full_action


class HeadlessMjInfer(MjInfer):
    def __init__(self, model_path, reference_data, onnx_model_path, standing=False):
        super().__init__(model_path, reference_data, onnx_model_path, standing)
        raw_leg_policy = self.policy
        self.policy = LegPolicyAdapter(raw_leg_policy, self)
        print("PPO actions: 10 leg joints")
        print("External head actions: actuator indices 5, 6, 7, 8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", required=True)
    parser.add_argument(
        "--reference_data",
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true")
    args = parser.parse_args()
    HeadlessMjInfer(
        args.model_path, args.reference_data, args.onnx_model_path, args.standing
    ).run()


if __name__ == "__main__":
    main()
