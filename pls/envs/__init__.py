from gymnasium.envs.registration import register


def register_envs() -> None:
    try:
        register(
            id="LineWorldSafety-v0",
            entry_point="pls.envs.line_world:LineWorldSafetyEnv",
        )
    except Exception:
        # Environment may already be registered when importing `pls` repeatedly.
        pass


__all__ = ["register_envs"]
