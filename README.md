# Differentiable Physics-Based Wheeled Robot Simulation

This repository contains the core training code, differentiable physics environment, and deployment scripts for our proposed **Diff-Wheelbot** model.

The policy is trained to output **local waypoints** (`model_mpc.py`), which a differentiable MPC (`mpc.py`) converts into linear/angular velocity commands that drive the differentiable CUDA physics simulator — forming a fully differentiable training loop (depth → waypoint policy → MPC → dynamics → loss).

## Prerequisites
Before running the simulation, you **must** have the following components installed and configured:

1. **[YOPO-Sim (Unity Simulator)](https://github.com/tju-aerial-robotics/yopo-sim)**: The core visual simulation environment.
2. **[ROS-TCP-Endpoint](https://github.com/Unity-Technologies/ROS-TCP-Endpoint)**: The essential communication bridge between ROS2 and Unity.

## Acknowledgements
We would like to express our sincere gratitude to the authors of the paper **"Back to Newton"** from **Shanghai Jiao Tong University (SJTU)**. Our differentiable physics implementation is heavily inspired by and modified upon their foundational open-source codebase. 

## Code Structure
```text
icra_code_0823
├── configs/
│   └── train_param.args     # Training hyperparameters (argparse @file format)
├── save/                    # Checkpoint directory (latest.pth + numbered checkpoints)
├── src/
│   ├── dynamics_kernel.cu   # Differentiable diff-drive robot dynamics kernels
│   ├── quadsim_kernel.cu    # Depth rendering & nearest-obstacle-point kernels
│   ├── quadsim.cpp          # CUDA to Python interface definition
│   ├── setup.py             # Build script for CUDA extensions
│   └── test.py              # Physics gradient testing script
├── env_cuda.py              # Differentiable simulation environment (PyTorch wrapper)
├── model_mpc.py             # Waypoint policy network (waypoints + desired speed)
├── model_direct.py          # Legacy direct-action policy (v, omega)
├── mpc.py                   # Differentiable waypoint MPC + perception helpers
├── train_mpc.py             # Waypoint + MPC training script (recommended)
├── train.py                 # Legacy direct-action training script
└── diff_wheelbot_ros2.py    # Model deployment script (ROS2)
```

## Environment Setup
The code is tested with the following environment:
- **PyTorch** 2.9.1 (cu130)
- **Python** 3.10.12
- **CUDA** 13.0
- **ROS2 Humble**

## Dependencies & Reproduction
To fully reproduce the model evaluation and visualize the simulation, you will need to set up the Unity simulator and the ROS2 communication bridge.

1. **YOPO-Sim (Unity Simulator)**: Required for visual evaluation. Please ensure you have the YOPO-Sim environment ready.
2. **ROS-TCP-Endpoint**: Required for the communication between our ROS2 node and the Unity simulator.

### 1. Compile CUDA Extensions
First, compile the differentiable physics CUDA kernels:
```bash
pip install -e src
# or
cd src 
python3 setup.py develop --user
```

### 2. Build & Run ROS2 TCP Endpoint
Clone and build the `ROS-TCP-Endpoint` package to establish the communication bridge:
```bash
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
cd ..
colcon build --packages-select ros_tcp_endpoint
source install/setup.bash
```
Launch the endpoint server (Ensure this is running before starting the Unity simulator):
```bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1 -p ROS_TCP_PORT:=10000
```

### 3. Training
All training hyperparameters (loss weights, model, MPC, perception, environment) are configurable in `configs/train_param.args`, which supports `#` comments; command-line arguments override the file values.
```bash
# Train the waypoint policy + MPC (uses configs/train_param.args by default)
python3 train_mpc.py @configs/train_param.args

# Override individual parameters on the command line
python3 train_mpc.py @configs/train_param.args --batch_size 32

# Resume from the latest checkpoint
python3 train_mpc.py @configs/train_param.args \
  --resume save/seed1_<timestamp>/latest.pth

# Legacy direct-action training
python3 train.py
```

For the matched static-versus-dynamic training ablation, keep all shared
hyperparameters in `train_param.args` and append exactly one overlay:

```bash
# Control: the existing fully static training distribution
python3 train_mpc.py \
  @configs/train_param.args \
  @configs/ablation_static.args \
  --seed 1

# Treatment: every scene contains moving obstacles; approximately 30% of
# cylinders/balls move at 0.2–1.0 m/s and reflect at the map boundary
python3 train_mpc.py \
  @configs/train_param.args \
  @configs/ablation_dynamic30.args \
  --seed 1
```

The overlay directories are experiment roots. Every new training process
automatically creates the same unique `seed<seed>_<timestamp>` child name below
its checkpoint and TensorBoard roots, so repeated or multi-seed runs do not
overwrite or merge. Pass `--run_name <name>` only when a stable custom child
name is useful. TensorBoard records
`Environment/DynamicSceneRate` and `Environment/MovingObstacleRate`; expected
values are `0/0` for the static group and approximately `1.0/0.30` for the
dynamic group. Boxes remain static and act as structural obstacles.
All training and benchmark scenes keep obstacle surfaces outside a 2 m radius
around the sampled start and goal. Moving obstacles reflect from these two
protected-zone boundaries, so they cannot occupy the goal during an episode.

After training, evaluate both checkpoints with the identical five-scene
protocol:

```bash
python3 benchmark.py \
  --checkpoint save/ablation_static/seed1_<timestamp>/latest.pth \
  --output-dir benchmark_results/ablation_static/seed1_<timestamp>

python3 benchmark.py \
  --checkpoint save/ablation_dynamic30/seed1_<timestamp>/latest.pth \
  --output-dir benchmark_results/ablation_dynamic30/seed1_<timestamp>
```

To resume, point `--resume` at a run checkpoint. Its original checkpoint and
TensorBoard directories are recovered from checkpoint metadata and reused:

```bash
python3 train_mpc.py @configs/train_param.args \
  @configs/ablation_static.args \
  --resume save/ablation_static/seed1_<timestamp>/latest.pth
```

Checkpoints are saved every `--ckpt_interval` iterations inside the unique run
directory (`checkpoint_mpc_<iter>.pth` plus `latest.pth`, containing model,
optimizer, scheduler, arguments and run metadata). TensorBoard logs include the
total loss, per-term losses, and success/collision/final-distance metrics.

Obstacle losses use differentiable signed clearance to the planar obstacle
footprint. The avoidance barrier keeps a base weight of one and adds a bounded,
detached approach-speed weight, so its spatial gradient always comes from
signed clearance and continues to point toward the nearest exit after
penetration.

For the four-run 10k local loss sweep (baseline, approach gain 1.25,
avoidance coefficient 0.9, and smoothness coefficient 0.10), run:

```bash
./scripts/run_loss_sweep_10k.sh 1
```

The argument is the training seed. Runs are sequential and write to
`save/loss_sweep_10k/` and `runs/loss_sweep_10k/`. Re-running the command skips
completed variants and resumes an incomplete variant from its `latest.pth`.
Set `RUN_SET` to a new label when intentionally creating another replicate
with the same seed.

State and odometry observation noise has been removed from the codebase to
reduce CPU overhead; training and benchmarking now always use exact-state
observations. Actuator domain randomization remains available through
`--max_action_delay_steps`, `--exec_v_scale_std`, `--exec_w_scale_std` and
`--wheel_bias_std`. The former observation-noise ablation scripts under
`scripts/` and their `observation_noise_*.args` overlays are kept only for
historical reference and no longer train distinct treatment arms.

### 4. Run the Simulation & Deployment
1. Start the **YOPO-Sim** Unity executable/editor and enter the simulation scene.
2. Once the ROS2 endpoint and Unity simulator are connected, open a new terminal and run the deployment script:
```bash
# To deploy and evaluate the trained model
python3 diff_wheelbot_ros2.py
```

### 5. Reproducible Benchmark

`benchmark.py` evaluates a checkpoint without gradient tracking using the same
depth → waypoint policy → MPC → actuator → CUDA dynamics pipeline as training.
The default protocol uses five evaluation seeds, 1024 episodes per seed and a
200-step horizon:

```bash
python3 benchmark.py \
  --checkpoint save/seed1_<timestamp>/latest.pth \
  --scenes open random dense cross dynamic \
  --seeds 0 1 2 3 4 \
  --episodes 1024 \
  --timesteps 200 \
  --output-dir benchmark_results/latest
```

The five scene definitions are:

- `open`: no obstacles;
- `random`: training-distribution density (20 cylinders, 12 balls, 10 boxes);
- `dense`: 1.5× obstacle counts (30 cylinders, 18 balls, 15 boxes);
- `cross`: random selection among all four corner-to-opposite-corner routes;
- `dynamic`: random-scene density, with 30% of cylinders/balls moving at
  0.2–1.0 m/s and reflecting at map boundaries.

Results are written to `summary.json` (full scene protocol, aggregate metrics,
95% Wilson intervals and per-seed metrics) and `episodes.csv` (one auditable row
per episode). Scene density and dynamic-obstacle parameters can be overridden
through the corresponding benchmark command-line flags.

Reported metrics are success rate, collision rate, SPL, command variation,
mean speed, final distance, minimum signed robot-body clearance and successful
arrival time. Success means entering the 1 m goal region before any collision;
collision means the robot's outer-surface signed clearance is no greater than
zero. Command variation is `mean((u[t] - u[t-1])**2)` and is not the training
smoothness loss.
