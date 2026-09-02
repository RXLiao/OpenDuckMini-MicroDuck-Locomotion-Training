"""Convert a 14-action Open Duck PPO checkpoint into a 10-action warm start."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.training import orbax_utils
from orbax import checkpoint as ocp


LEG_ACTION_IDS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)


def restore(path: Path):
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(str(path))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    restore_args = jax.tree.map(
        lambda _: ocp.ArrayRestoreArgs(sharding=sharding), metadata
    )
    return checkpointer.restore(str(path), restore_args=restore_args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = restore(Path(args.input).resolve())
    output_ids = jnp.array(
        LEG_ACTION_IDS + tuple(i + 14 for i in LEG_ACTION_IDS), dtype=jnp.int32
    )
    actor = source[1]
    last = actor["params"]["hidden_3"]
    if last["kernel"].shape[-1] != 28 or last["bias"].shape[-1] != 28:
        raise ValueError("source actor is not a 14-action PPO checkpoint")
    actor["params"]["hidden_3"]["kernel"] = last["kernel"][:, output_ids]
    actor["params"]["hidden_3"]["bias"] = last["bias"][output_ids]

    target = [source[0], actor, source[2]]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(
        str(output),
        target,
        force=True,
        save_args=orbax_utils.save_args_from_target(target),
    )
    print("OUTPUT", output)
    print("ACTOR_OUTPUT", target[1]["params"]["hidden_3"]["bias"].shape)


if __name__ == "__main__":
    main()
