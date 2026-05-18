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

## Installation

Quick install if you already know what you are doing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
cd simulator && python setup.py build_ext --inplace && cd ..
python -m pytest
```

### CachyOS step-by-step installation guide

CachyOS is based on Arch Linux, so the commands below use `pacman`. Copy and
paste one command block at a time. If a command asks for your password, type
your normal user password and press Enter.

#### 1. Open a terminal

Open the CachyOS application launcher, search for **Terminal**, and open it.
All commands below go into that terminal window.

#### 2. Update CachyOS first

This makes sure your system packages are current before Python packages are
installed.

```bash
sudo pacman -Syu
```

If CachyOS asks whether to proceed, press `Y` and then Enter. If the update
installs a new kernel or a lot of system packages, reboot before continuing:

```bash
sudo reboot
```

After the reboot, open Terminal again.

#### 3. Install the system tools this project needs

Install Git, Python, pip, virtual environment support, and build tools for the
Cython simulator extension.

```bash
sudo pacman -S --needed git python python-pip python-virtualenv base-devel
```

When CachyOS shows a package list, press Enter to accept the defaults, then
press `Y` and Enter if it asks for confirmation.

#### 4. Download the project

Choose a folder where you keep projects. This example uses `~/Projects`.

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/giorgiopulcini/Haxball-AI.git
cd Haxball-AI
```

If you are installing from your own fork, replace the `git clone` URL with your
fork URL. The important part is that your terminal must be inside the project
folder before you continue. You can check that with:

```bash
pwd
```

The output should end with `Haxball-AI`.

#### 5. Create a private Python environment for this project

Do not install the Python packages globally. A virtual environment keeps this
project's packages separate from the rest of your CachyOS system.

```bash
python -m venv .venv
```

#### 6. Activate the virtual environment

```bash
source .venv/bin/activate
```

After this, your terminal prompt should start with `(.venv)`. That means the
virtual environment is active. If you close the terminal later, come back to the
project folder and run this same `source .venv/bin/activate` command again
before using the project.

#### 7. Upgrade Python packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

#### 8. Install the project dependencies

```bash
python -m pip install -r requirements.txt
```

This can take a while because PyTorch, Stable-Baselines3, Gymnasium, pygame,
and the scientific Python packages are not tiny. Wait until the command finishes
and returns you to the terminal prompt.

#### 9. Build the optional fast simulator extension

The project can use a Cython simulator extension. Build it from inside the
`simulator` folder, then return to the project root.

```bash
cd simulator
python setup.py build_ext --inplace
cd ..
```

#### 10. Check that the installation works

Run the test suite:

```bash
python -m pytest
```

You want to see the tests finish without failures. Warnings are usually okay,
but red `FAILED` lines mean something went wrong.

You can also run Stable-Baselines3's environment checker with a tiny training
run:

```bash
python scripts/train_ppo.py --check-env --total-timesteps 1000
```

#### 11. Start a training run

Once the checks pass, start training. The example below writes the model to
`models/haxball_ppo.zip`.

```bash
python scripts/train_ppo.py --total-timesteps 1000000 --output models/haxball_ppo
```

Training may take a long time. Start with fewer timesteps, such as `10000`, if
you only want to confirm that training starts correctly.

#### 12. Play interactively after you have a model

When training has created `models/haxball_ppo.zip`, run:

```bash
python openai_interactive.py --model models/haxball_ppo.zip
```

If you do not have a trained model yet, use the random fallback instead:

```bash
python openai_interactive.py --fallback random
```

#### Troubleshooting on CachyOS

* **`python: command not found`**: run
  `sudo pacman -S --needed python`.
* **`pip` refuses to install globally**: that is expected on modern Arch-based
  systems. Activate the virtual environment with
  `source .venv/bin/activate`, then use `python -m pip ...`.
* **`error: command 'gcc' failed` or missing compiler tools**: run
  `sudo pacman -S --needed base-devel`, then rebuild the simulator extension.
* **`ModuleNotFoundError` for a package from `requirements.txt`**: make sure
  `(.venv)` appears in your prompt, then run
  `python -m pip install -r requirements.txt` again.
* **pygame opens a blank window or no window**: make sure you are running from
  a normal graphical CachyOS desktop session, not from a TTY-only console or an
  SSH session without display forwarding.
* **You closed the terminal and commands stopped working**: go back to the
  project folder and reactivate the environment:

  ```bash
  cd ~/Projects/Haxball-AI
  source .venv/bin/activate
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
