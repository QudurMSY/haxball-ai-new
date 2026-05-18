# Haxball-AI

![Bot-vs-Human](docs/bot_vs_human.gif)

* [Talk on Machine Learning Milan event](https://youtu.be/Ma_MopOXLCg)
* [Full gameplay demostration video](https://youtu.be/fpIe6lNM1NE)

This is an implementation of a bot for [haxball](https://www.haxball.com/) using reinforcement learning.

*Attention! This repo does not provide any connector to the browser!*

## Modern stack

This fork targets the modern Gymnasium API and Stable-Baselines3 instead of the old TensorFlow 1.x/OpenAI Baselines stack.

Core dependencies:

* `gymnasium`
* `stable-baselines3`
* `pygame`
* `numpy`
* `cython`
* `torch`
* `tqdm`
* `pytest`
* `pandas` (for existing simulator tests)

The environment is exposed as `haxball_ai.env.HaxballEnv` and still uses the simulator in `simulator/`.

## Full installation guide for CachyOS

CachyOS is Arch-based, so the commands below use `pacman`. Run them one by one from a terminal.

### 1. Install system packages

```bash
sudo pacman -Syu
sudo pacman -S --needed git base-devel uv python python-pip sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

What these are for:

* `git` downloads the project.
* `base-devel` gives you compilers for packages with native code.
* `uv` creates a clean Python environment and can install a supported Python version.
* `sdl2*` packages help `pygame` open a game window.

### 2. Download this repository

If you already have the repository on your computer, skip this step and `cd` into your existing folder.

```bash
git clone <REPLACE_WITH_THIS_REPOSITORY_URL> haxball-ai
cd haxball-ai
```

If you downloaded a ZIP instead, extract it, then open a terminal in the extracted folder.

### 3. Create a clean Python environment

Use Python 3.11 because it is a safe choice for PyTorch, Stable-Baselines3, Gymnasium, and pygame.

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
```

After activation, your terminal prompt should usually start with `(.venv)`. If it does not, run this again:

```bash
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Do **not** use `sudo pip install ...` inside the project. Keep everything inside `.venv`.

### 5. Check that the environment imports correctly

```bash
python - <<'PY'
from haxball_ai.env import HaxballEnv

env = HaxballEnv(max_ticks=10)
obs, info = env.reset(seed=0)
print('obs shape:', obs.shape)
print('obs dtype:', obs.dtype)
print('reset info:', info)
PY
```

Expected result: it prints `obs shape: (14,)` and `obs dtype: float32`.

### 6. Optional: build the Cython simulator extension

The normal scripts can use the pure-Python simulator. Build this only if you specifically want the optional Cython extension.

```bash
cd simulator
python setup.py build_ext --inplace
cd ..
```

### 7. Run the tests

```bash
python -m pytest
```

If this fails because a package is missing, make sure your virtual environment is active and rerun:

```bash
python -m pip install -r requirements.txt
```

### 8. Run a tiny training smoke test

This checks the Gymnasium/Stable-Baselines3 integration without waiting for a long training run.

```bash
python scripts/train_ppo.py --check-env --total-timesteps 1000 --output models/haxball_ppo_smoke
```

Expected result: a model file appears at `models/haxball_ppo_smoke.zip`.

### 9. Train a real model

This can take a while depending on your CPU/GPU.

```bash
python scripts/train_ppo.py --total-timesteps 1000000 --output models/haxball_ppo
```

Expected result: a model file appears at `models/haxball_ppo.zip`.

### 10. Play against the model

```bash
python openai_interactive.py --model models/haxball_ppo.zip
```

Controls:

* Arrow keys move your player.
* Space kicks.
* Close the pygame window to quit.

If you have not trained a model yet, play against a random fallback opponent:

```bash
python openai_interactive.py --fallback random
```

### Troubleshooting on CachyOS

#### `uv: command not found`

Install it:

```bash
sudo pacman -S --needed uv
```

#### `ModuleNotFoundError`

You are probably outside the virtual environment. From the repository root, run:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

#### Pygame opens no window or crashes on Wayland

Try forcing SDL to use X11:

```bash
SDL_VIDEODRIVER=x11 python openai_interactive.py --fallback random
```

#### PyTorch install problems

First try upgrading pip inside the virtual environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use an NVIDIA GPU and want a specific CUDA build, follow the official PyTorch install command for your driver/CUDA version, then rerun:

```bash
python -m pip install -r requirements.txt
```


## Gymnasium API

`HaxballEnv` follows Gymnasium's current API:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

The action space is `Discrete(10)`. The observation space is a 14-value `Box` with `np.float32` observations.

Episode endings are split as follows:

* `terminated=True` for goals and start-timeout failures.
* `truncated=True` when the configured `max_ticks` time limit is reached.

## Training

Train PPO with Stable-Baselines3:

```bash
python scripts/train_ppo.py --total-timesteps 1000000 --output models/haxball_ppo
```

You can run Stable-Baselines3's environment checker before training:

```bash
python scripts/train_ppo.py --check-env --total-timesteps 1000
```

## Playing interactively

After saving a Stable-Baselines3 PPO checkpoint, run:

```bash
python openai_interactive.py --model models/haxball_ppo.zip
```

If the model file does not exist, the script can fall back to a static or random opponent:

```bash
python openai_interactive.py --fallback random
```

## Legacy code

The previous OpenAI Baselines/TensorFlow 1.x training implementation was intentionally not ported. `hx_controller/openai_model_torneo.py` now raises a migration error that points users to Stable-Baselines3 PPO.
