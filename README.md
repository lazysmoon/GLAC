# GLAC: Graph-based Lyapunov Actor-Critic

Official JAX implementation of the paper:

**Safe Control in Obstacle-Cluttered Environments via Graph-based Lyapunov Reinforcement Learning**

GLAC learns a safe control policy for navigation in obstacle-cluttered environments by combining graph-based state representation with Lyapunov-constrained reinforcement learning.

## 📁 Project Structure

```
GLAC/
├── glac/               # Core implementation (agent, environments, utils)
├── pretrain/           # Pretrained checkpoints for evaluation
├── train.py            # Training entry
├── evaluate.py         # Evaluation entry
├── requirements.txt    # Dependencies
└── README.md
```

## 🔧 Installation

```bash
git clone https://github.com/<your-username>/GLAC.git
cd GLAC

conda create -n glac python=3.10 -y
conda activate glac

pip install -r requirements.txt
```

> **Note on JAX:** If you want GPU acceleration, please install the matching JAX build for your CUDA version following the [official JAX installation guide](https://github.com/google/jax#installation). The `requirements.txt` installs the CPU version by default.

## 🚀 Usage

### Training

Train a GLAC agent from scratch with default parameters:

```bash
python train.py
```

Common options:

```bash
python train.py \
    --env Second_order \
    --obs 16 \
    --area-size 4 \
    --seed 0
```

Training logs and checkpoints will be saved under `./logs/<env>/<algo>/seed<seed>_<timestamp>/`.

### Evaluation

Evaluate a pretrained model (shipped under `pretrain/`):

```bash
python evaluate.py --model_dir ./pretrain --epi 100
```

The script will report the mean return, success rate, and safe rate, and save rendered trajectories (`.png`) and videos (`.mp4`) of sampled episodes to `./pretrain/eval_obs<obs>_<prefix>/`.

Common options:

| Argument | Description | Default |
|---|---|---|
| `--model_dir` | Directory containing the checkpoint | `./pretrain` |
| `--prefix` | Checkpoint filename prefix | `checkpoint` |
| `--epi` | Number of evaluation episodes | `100` |
| `--obs` | Number of obstacles in the environment | `20` |
| `--max_step` | Maximum steps per episode | `320` |
| `--seed` | Random seed | `123` |



## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
