# EngineAI-Lab

> A standalone MuJoCo MJX + JAX/ROCm training path (without Isaac Lab) is
> documented in [docs/MJX_TRAINING.md](docs/MJX_TRAINING.md).

**EngineAI Lab** is a python package for training and deploying policies for EngineAI Robots using Isaac Lab and Isaac Sim.

|Training| Sim2Sim |Deploy|
|--------|--------|--------|
![training](./docs/training.gif)|![sim2sim](./docs/sim2sim.gif)|<img src="./docs/deploy.gif" height="180"/>

# Structure

```
engineai-lab
├── config
├── dataset
│   ├── config
│   └── data
├── scripts
└── source
    └── engineai_lab
        ├── algorithms      
        ├── assets
        │   └── pm01
        │       ├── meshes
        │       └── urdf
        ├── robots
        ├── tasks
        │   └── velocity
        │       ├── config
        │       │   └── pm01
        │       └── mdp
        └── utils
```

## QUICKSTART

### 1. Create a Conda Environment

Create and activate a new environment with Python 3.11:

  ```bash
  conda create -n engineai_lab python=3.11
  conda activate engineai_lab
  ```

### 2. Install Prerequisites

- **Install Isaac Sim**

  Follow the official installation guide: [Isaac Lab - Pip Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-dependencies).

    *Since you've already created the engineai_lab environment, follow the guide from "Installing Dependencies" up to (but not including) the "Installing Isaac Lab" section.*

- **Clone & Setup Isaac Lab**

  Clone the repository and switch to the recommended branch:

  ```bash
  git clone https://github.com/isaac-sim/IsaacLab.git
  cd IsaacLab
  git checkout 4df6560e
  ./isaaclab -i rsl_rl   # Install rsl-rl dependency
  ```

We highly recommend using the main branch`(4df6560e)` of Isaac Lab, as it can support rsl-rl-lib >= 5.0 and Isaac Sim >= 5.0 .

### 3. Install this Package

Once the prerequisites are set up, install the package in editable mode:

```bash
pip install -e .
```

## Usage

### Supported Robots

This repository currently supports the following environments from the EngineAI Robots family:

|Robot| Task |Description|
|--------|--------|--------|
PM01|`Flat-PM01-v0`|Basic flat-terrain locomotion
PM01|`Flat-AMP-PM01-v0`|AMP-based motion imitation on flat terrain

*More robots and environments are coming soon!*

### Training a Policy

```
 python scripts/train.py --task=Flat-PM01-v0  --num_envs 4096 --headless --run_name <name>  
 python scripts/play.py  --task=Flat-PM01-v0 --num_envs 128 --load_run <name> 
```

### Evaluating a Policy

```
 python scripts/train.py --task=Flat-AMP-PM01-v0  --num_envs 4096 --headless --run_name <name> 
 python scripts/play.py --task=Flat-AMP-PM01-v0 --num_envs 128 --load_run <name>
```

Replace `<name>` with the name of your training run (found in logs/rsl_rl/).

### Deployment

To deploy a trained policy on real hardware, convert it to the MNN format for efficient inference.

#### 1. Export PyTorch Policy to ONNX

(Ensure your training script supports ONNX export)

#### 2. Build [MNN-Converter](https://mnn-docs.readthedocs.io/en/latest/start/quickstart_cpp.html?highlight=mnn+converter)

```bash
git clone https://github.com/alibaba/mnn
cd mnn
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j8
```

#### 3. Convert ONNX into MNN

```bash
./MNNConvert -f ONNX \
  --modelFile path_to_your_policy.onnx \
  --MNNModel your_policy.mnn \
  --bizCode MNN
```

For detailed instructions on integrating the MNN model with EngineAI robots, see:
[engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk).

## Support

If you have any questions about using this repository, we're here to help!

- **Report Issues**: Found a bug or have a feature request. Please open a new issue on our [GitHub Issues](https://github.com/engineai-robotics/engineai_lab/issues) page.
- **Email Us**: For general inquiries or collaboration opportunities, feel free to reach out at [info@engineai.com.cn](mailto:info@engineai.com.cn).

## License

EngineAI-Lab is released under [BSD-3 License](LICENSE).

## Acknowledgement

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [**IsaacLab**](https://github.com/isaac-sim/IsaacLab) — The foundational framework for training and running simulation experiments.
- [**rsl_rl**](https://github.com/leggedrobotics/rsl_rl) — High-performance reinforcement learning library for legged robots.
- [**AMP_for_hardware**](https://github.com/escontra/AMP_for_hardware) — Implementation of Adversarial Motion Priors (AMP) for sim-to-real transfer.
- [**BeyondMimic**](https://github.com/HybridRobotics/whole_body_tracking) — Inspiration for project structure and valuable feature implementations.
- [**MNN**](https://github.com/alibaba/mnn) — Lightweight, high-performance inference engine for on-device deployment.
- [**engineai_robotics_native_sdk**](https://github.com/engineai-robotics/engineai_robotics_native_sdk) — Official SDK for deploying policies on EngineAI robotic hardware.
