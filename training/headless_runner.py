"""Train the 10-action Open Duck Mini locomotion policy."""

import argparse
import functools

from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from flax import linen
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import headless_joystick


class HeadlessRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        self.env_config = headless_joystick.default_config()
        self.env = headless_joystick.HeadlessJoystick(task=args.task)
        self.eval_env = headless_joystick.HeadlessJoystick(task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])
        self.restore_checkpoint_path = args.restore_checkpoint_path

    def train(self) -> None:
        config = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )
        params = dict(config)
        params.pop("network_factory", None)
        from_scratch = self.restore_checkpoint_path is None
        params.update(
            num_timesteps=self.num_timesteps,
            num_envs=8192,
            num_evals=24,
            num_resets_per_eval=1,
            episode_length=1000,
            batch_size=8192,
            num_minibatches=2,
            unroll_length=20,
            num_updates_per_batch=2,
            # Random initialization needs substantially more exploration and
            # a normal PPO learning rate.  Keep conservative fine-tune values
            # only when an explicit checkpoint is supplied.
            learning_rate=3e-4 if from_scratch else 5e-5,
            entropy_cost=0.01 if from_scratch else 0.001,
            normalize_observations=True,
            discounting=0.99,
            seed=20260831,
            deterministic_eval=True,
        )
        print("action_size:", self.action_size, flush=True)
        print("obs_size:", self.obs_size, flush=True)
        print("PPO params:", params, flush=True)

        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            activation=linen.swish,
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )

        train_fn = functools.partial(
            ppo.train,
            **params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.save_checkpoint_only_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )
        train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        print("Training finished.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_timesteps", type=int, default=150_000_000)
    parser.add_argument("--task", default="flat_terrain")
    parser.add_argument(
        "--restore_checkpoint_path",
        default=None,
        help="optional PPO checkpoint; omit to train from random initialization",
    )
    args = parser.parse_args()
    HeadlessRunner(args).train()


if __name__ == "__main__":
    main()
