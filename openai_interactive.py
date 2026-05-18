"""Play Haxball against a Stable-Baselines3 PPO policy.

The old version of this script loaded TensorFlow 1.x/OpenAI Baselines checkpoints.
Modern checkpoints should be saved with Stable-Baselines3, for example as
``models/haxball_ppo.zip``.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
from stable_baselines3 import PPO

from hx_controller.haxball_gym import Haxball
from simulator.simulator.cenv import Vector as CVector
from simulator.simulator.cenv import create_start_conditions as Ccreate_start_conditions
from simulator.visualizer import draw_frame


DEFAULT_MODEL_PATH = Path("models/haxball_ppo.zip")


class SB3Model:
    """Small adapter that exposes a single-action ``predict`` API."""

    def __init__(self, model_path: Path, deterministic: bool = True) -> None:
        self.model_path = model_path
        self.deterministic = deterministic
        self.model = PPO.load(model_path)

    def predict(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(np.asarray(action).item())


class StaticActionModel:
    """Fallback model that always returns the configured action."""

    def __init__(self, default_action: int = 0) -> None:
        self.default_action = default_action

    def predict(self, obs: np.ndarray) -> int:
        return self.default_action


class RandomActionModel:
    """Fallback model that samples directly from the environment action space."""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def predict(self, obs: np.ndarray) -> int:
        return int(self.action_space.sample())


class DelayedModel:
    """Poll the Gymnasium env, predict an action, then apply it after a delay."""

    def __init__(self, env: Haxball, model, play_red: bool) -> None:
        self.state = 0
        self.env = env
        self.model = model
        self.play_red = play_red
        self.wait_time = 2
        self.obs: Optional[np.ndarray] = None
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.action = 0

    def gameplay_tick(self):
        if self.state == 0:
            self.obs, self.reward, self.terminated, self.truncated, self.info = self.env.step_wait(
                red_team=not self.play_red
            )
            if self.terminated or self.truncated:
                self.obs, _ = self.env.reset()
            self.state = 2
            self.wait_time = 0

        elif self.state == 1:
            if self.wait_time == 0:
                self.state = 2
            self.wait_time -= 1

        elif self.state == 2:
            self.action = self.model.predict(self.obs)
            self.state = 4
            self.wait_time = 0

        elif self.state == 3:
            if self.wait_time == 0:
                self.state = 4
            self.wait_time -= 1

        elif self.state == 4:
            self.env.step_async(self.action, red_team=not self.play_red)
            self.state = 0
            self.wait_time = 5

        elif self.state == 5:
            if self.wait_time == 0:
                self.state = 0
            self.wait_time -= 1


def build_ai_model(model_path: Path, fallback: str, action_space):
    if model_path.exists():
        return SB3Model(model_path)
    if fallback == "random":
        return RandomActionModel(action_space)
    return StaticActionModel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play against a Stable-Baselines3 Haxball PPO agent.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to a SB3 PPO .zip model.")
    parser.add_argument("--play-red", action="store_true", help="Control the red player instead of blue.")
    parser.add_argument("--fallback", choices=("static", "random"), default="static", help="AI used if --model is absent.")
    parser.add_argument("--max-ticks", type=int, default=int(60 * 3 * (1 / 0.016)) * 2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((900, 520))

    gameplay = Ccreate_start_conditions(
        posizione_palla=CVector(0, 0),
        velocita_palla=CVector(0, 0),
        posizione_blu=CVector(277.5, 0),
        velocita_blu=CVector(0, 0),
        input_blu=0,
        posizione_rosso=CVector(-277.5, 0),
        velocita_rosso=CVector(0, 0),
        input_rosso=0,
        tempo_iniziale=0,
        punteggio_rosso=0,
        punteggio_blu=0,
    )

    env = Haxball(gameplay=gameplay, max_ticks=args.max_ticks)
    env.reset()
    ai_model = build_ai_model(args.model, args.fallback, env.action_space)
    delayed_model = DelayedModel(env, ai_model, args.play_red)

    human_write_index = 1 if args.play_red else 2
    blue_unpressed = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        gameplay.Pa.D[human_write_index].mb = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            gameplay.Pa.D[human_write_index].mb |= 1
        if keys[pygame.K_DOWN]:
            gameplay.Pa.D[human_write_index].mb |= 2
        if keys[pygame.K_RIGHT]:
            gameplay.Pa.D[human_write_index].mb |= 8
        if keys[pygame.K_LEFT]:
            gameplay.Pa.D[human_write_index].mb |= 4
        if keys[pygame.K_SPACE]:
            if blue_unpressed:
                gameplay.Pa.D[human_write_index].mb |= 16
                gameplay.Pa.D[human_write_index].bc = 1
            blue_unpressed = False
        else:
            gameplay.Pa.D[human_write_index].bc = 0
            blue_unpressed = True

        delayed_model.gameplay_tick()
        draw_frame(screen, gameplay, reward=delayed_model.reward, ret=delayed_model.info.get("score"))
        pygame.display.flip()
        clock.tick(60)
        gameplay.step(1)


if __name__ == "__main__":
    main()
