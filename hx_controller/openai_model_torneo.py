"""Legacy OpenAI Baselines/TensorFlow torneo model placeholder.

This project now uses Gymnasium environments and Stable-Baselines3 PPO instead of
porting the old TensorFlow 1.x/OpenAI Baselines implementation.  Keeping this
module lightweight avoids importing deprecated TensorFlow/Baselines dependencies
from modern code while still giving old imports a clear migration error.
"""


class A2CModel:
    """Deprecated compatibility shim for the removed Baselines model."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "hx_controller.openai_model_torneo.A2CModel was removed during the "
            "Gymnasium/Stable-Baselines3 migration. Train or load a "
            "stable_baselines3.PPO model instead."
        )
