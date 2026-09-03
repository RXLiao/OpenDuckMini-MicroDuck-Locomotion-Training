# Open Duck Mini — Headless 10D Locomotion

This repository contains the 10-action locomotion policy trained for Open Duck Mini.
The PPO policy controls only the ten leg actuators. Head joints remain in the
101-dimensional observation and are controlled externally by the head-command
mode in the runtime.

## Demo videos

### Real-world test

https://github.com/user-attachments/assets/c6ff4f49-1e8a-4cb6-8dd5-1342ca919889

### MuJoCo simulation

https://github.com/user-attachments/assets/69d749e8-d26b-4c3c-b6f1-05ff4a343198

## Contents

- `HEADLESS_WALK_10D_300M.onnx` — exported policy; input `(1, 101)`, output `(1, 10)`.
- `v2_rl_walk_headless.py` — Raspberry Pi deployment script with one-second command ramp.
- `mujoco_infer_headless.py` — MuJoCo inference adapter for the 10D policy.
- `headless_joystick.py` — MuJoCo/JAX environment that maps 10 leg actions to 14 actuators.
- `headless_runner.py` — PPO training entry point.
- `export_headless_onnx.py` — checkpoint-to-ONNX exporter and validator.
- `convert_checkpoint_14_to_10.py` — converts a 14-action PPO checkpoint to a 10-action warm start.
- `eval_headless_onnx.py` — evaluator entry point for the headless environment.

## Deployment

Copy the ONNX file and `v2_rl_walk_headless.py` into the robot runtime `scripts`
directory and run:

```bash
workon open-duck-mini-runtime
python v2_rl_walk_headless.py \
  --onnx_model_path ./HEADLESS_WALK_10D_300M.onnx \
  --duck_config_path /home/lxkj/duck_config.json \
  --command_ramp_duration 1.0
```

The script expects the existing `control_server.py` and `XBoxController` command
service. It accepts the seven-command vector
`[vx, vy, yaw, neck_pitch, head_pitch, head_yaw, head_roll]`.

## Training

Training uses JAX, MuJoCo/MJX, Brax PPO and ROCm. On a two-GPU ROCm host, the
RCCL library may need compatibility links named `libnccl.so` and `libnccl.so.2`.
The supplied runner supports random initialization when no restore checkpoint is
given, or warm-start conversion from an existing 14-action checkpoint.

## Model selection

The included ONNX was selected using repeated 50/100-episode evaluations. It is
an experimental research model; test with the robot suspended and retain a safe
stop procedure before walking on the floor.

## License and attribution

This project builds on Open Duck Mini and MuJoCo/Brax ecosystem code. Check the
upstream project licenses before redistributing upstream assets.
