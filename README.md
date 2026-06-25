# GLAC: Graph-based Lyapunov Actor-Critic

Official JAX implementation of the paper:

**Safe Control for Quadrotors in Cluttered Environments via Graph-based Lyapunov Reinforcement Learning**

GLAC learns a safe control policy for navigation in obstacle-cluttered environments by combining a graph-based state representation with Lyapunov-constrained reinforcement learning.

## 📁 Project Structure

```
GLAC/
├── glac/               # Core implementation (agent, environments, networks, utils)
├── pretrain/           # Pretrained checkpoint and its config for evaluation
├── train.py            # Training entry
├── evaluate.py         # Evaluation entry
├── requirements.txt    # Dependencies
└── README.md
```

## 🔧 Installation

```bash
git clone https://github.com/lazysmoon/GLAC.git
cd GLAC

conda create -n glac python=3.10 -y
conda activate glac

pip install -r requirements.txt
```

> **Note on JAX:** For GPU acceleration, install the JAX build matching your CUDA version by following the [official JAX installation guide](https://github.com/google/jax#installation).

## 🚀 Usage

### Training

Train a GLAC agent from scratch with default parameters:

```bash
python train.py
```

Common options:

```bash
python train.py \
    --env DubinsCar \
    --obs 8 \
    --area-size 6 \
    --seed 52
```

Training logs and checkpoints are saved under `./logs/<env>/<algo>/seed<seed>_<timestamp>/`.

### Evaluation

Evaluate the pretrained model shipped under `pretrain/` (loaded directly from `pretrain/checkpoint`, with its `config.yaml` read from `pretrain/`):

```bash
python evaluate.py
```

The script reports the mean return, success rate, and safe rate, and saves the
summary (`output.txt`) together with rendered trajectory plots (`.png`) of
sampled episodes to `./pretrain/eval_obs<obs>_checkpoint/`.

Common options:

| Argument | Description | Default |
|---|---|---|
| `--model_dir` | Path to the checkpoint directory | `./pretrain/checkpoint` |
| `--prefix` | Checkpoint name prefix (only used when `model_dir` holds multiple checkpoints) | `checkpoint_` |
| `--epi` | Number of evaluation episodes | `100` |
| `--obs` | Number of obstacles in the environment | `8` |
| `--max_step` | Maximum steps per episode | `256` |
| `--seed` | Random seed | `123` |

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
