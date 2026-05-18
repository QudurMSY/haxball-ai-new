"""Train a Stable-Baselines3 PPO policy on the Haxball Gymnasium env."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from haxball_ai.env import HaxballEnv


def make_env(max_ticks: int):
    return Monitor(HaxballEnv(max_ticks=max_ticks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stable-Baselines3 PPO for Haxball.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("models/haxball_ppo"))
    parser.add_argument("--check-env", action="store_true", help="Run SB3's custom env checker before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(args.max_ticks)
    if args.check_env:
        check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.total_timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()
