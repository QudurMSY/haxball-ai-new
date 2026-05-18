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

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the optional Cython simulator extension:

```bash
cd simulator && python setup.py build_ext --inplace
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
