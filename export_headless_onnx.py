"""Export and validate a 101-input, 10-output Open Duck PPO checkpoint."""

from __future__ import annotations

import os
os.environ.pop("XLA_FLAGS", None)
os.environ.pop("HIP_VISIBLE_DEVICES", None)
os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from pathlib import Path

import jax
import numpy as np
import onnx
import onnxruntime as ort
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp

from playground.common.export_onnx import export_onnx


def restore(path: Path):
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(str(path))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    restore_args = jax.tree.map(
        lambda _: ocp.ArrayRestoreArgs(sharding=sharding), metadata
    )
    return checkpointer.restore(str(path), restore_args=restore_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    params = restore(Path(args.checkpoint).resolve())
    config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    output = Path(args.output).resolve()
    export_onnx(params, 10, config, 101, output_path=str(output))

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    result = session.run(
        None, {input_meta.name: np.zeros((1, 101), dtype=np.float32)}
    )[0]
    if result.shape != (1, 10) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"invalid ONNX output: {result.shape}")
    print("INPUT", input_meta.shape, input_meta.type)
    print("OUTPUT", result.shape)
    print("FILE", output, output.stat().st_size)


if __name__ == "__main__":
    main()
