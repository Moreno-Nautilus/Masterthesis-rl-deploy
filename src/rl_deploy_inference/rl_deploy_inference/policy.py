"""rl_games actor adapter for Isaac-free deployment."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import yaml


class PolicyLoadError(RuntimeError):
    pass


def _find_insertion_policy_base(train_repo: str) -> Path | None:
    candidates = [
        Path(train_repo),
        Path(train_repo) / "source" / "insertion_policy",
        Path(train_repo).parent / "source" / "insertion_policy",
    ]
    for path in candidates:
        if (path / "insertion_policy").is_dir():
            return path
    return None


def _register_insertion_network(train_repo: str) -> None:
    """Register the ``insertion_hybrid`` rl_games network for an Isaac-free deploy.

    The network registration is a single ``model_builder.register_network(...)`` call at the bottom of
    ``insertion_hybrid_network.py`` and needs only torch + rl_games. We exec THAT FILE directly instead
    of ``import insertion_policy.tasks.insertion.agents``, because the package ``__init__`` chain pulls
    in ``isaaclab`` -> ``pxr``/USD (the Isaac Sim runtime), which is exactly what this deploy avoids.
    Falls back to the full package import if the file layout is unexpected.
    """
    from rl_games.algos_torch import model_builder

    if "insertion_hybrid" in model_builder.NETWORK_REGISTRY:
        return
    base = _find_insertion_policy_base(train_repo)
    if base is not None:
        net_file = (
            base / "insertion_policy" / "tasks" / "insertion" / "agents" / "insertion_hybrid_network.py"
        )
        if net_file.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("insertion_hybrid_network_deploy", net_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # runs register_network("insertion_hybrid", ...)
            if "insertion_hybrid" in model_builder.NETWORK_REGISTRY:
                return
    # Fallback: full package import (requires the whole training env, including isaaclab/pxr).
    if base is not None and str(base) not in sys.path:
        sys.path.insert(0, str(base))
    import insertion_policy.tasks.insertion.agents  # noqa: F401


def _patch_rl_games_dict_rms() -> None:
    import torch
    from rl_games.algos_torch.running_mean_std import RunningMeanStdObs

    if getattr(torch.jit, "_insertion_rmsobs_patch", False):
        return
    orig_jit_script = torch.jit.script

    def _jit_script_skip_rmsobs(obj, *args, **kwargs):
        if isinstance(obj, RunningMeanStdObs):
            return obj
        return orig_jit_script(obj, *args, **kwargs)

    torch.jit.script = _jit_script_skip_rmsobs
    torch.jit._insertion_rmsobs_patch = True


class RlGamesActor:
    """Loads the saved rl_games player and exposes only actor actions."""

    def __init__(
        self,
        checkpoint: str,
        agent_config: str,
        train_repo: str,
        device: str,
        deterministic: bool,
        policy_dim: int = 21,
        image_shape: tuple[int, int, int] = (224, 224, 4),
        action_dim: int = 5,
    ):
        checkpoint = os.path.expanduser(checkpoint)
        agent_config = os.path.expanduser(agent_config)
        if not os.path.isfile(checkpoint):
            raise PolicyLoadError(f"Policy checkpoint not found: {checkpoint}")
        if not os.path.isfile(agent_config):
            raise PolicyLoadError(f"rl_games agent config not found: {agent_config}")

        try:
            import gymnasium as gym
            import torch
            from rl_games.torch_runner import Runner
        except ImportError as exc:
            raise PolicyLoadError(
                "torch/gymnasium/rl_games are required for policy inference. "
                "Run the node from the Isaac/rl_games Python environment."
            ) from exc

        try:
            _register_insertion_network(train_repo)
        except Exception as exc:  # noqa: BLE001
            raise PolicyLoadError(
                "Could not register the training repo's insertion_hybrid network. "
                f"train_repo={train_repo!r}"
            ) from exc
        _patch_rl_games_dict_rms()

        with open(agent_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = copy.deepcopy(cfg)
        params = cfg["params"]
        params["load_checkpoint"] = True
        params["load_path"] = checkpoint

        # Explicit-estimator checkpoints (e.g. w2_estimator_192) list a privileged `aux_label` obs
        # group and an `aux_head` in the network. `aux_label` is a TRAINING-ONLY label (the true hole
        # gap); the network excludes it from the policy input and skips the aux loss at inference, so it
        # has ZERO effect on the action. BUT `normalize_input` saved a RunningMeanStdObs sub-module for
        # `aux_label`, so the obs space MUST declare it or the checkpoint restore mismatches. We add the
        # key here and feed dummy zeros each step (see act()). Auto-detected from the agent config so a
        # non-estimator checkpoint (e.g. e2e_weld_curric, no aux_head) leaves the obs space unchanged.
        self._aux_label_key, self._aux_label_dim = self._detect_aux_label(params)

        obs_dict = {
            "policy": gym.spaces.Box(-np.inf, np.inf, shape=(policy_dim,), dtype=np.float32),
            "image": gym.spaces.Box(-np.inf, np.inf, shape=image_shape, dtype=np.float32),
        }
        if self._aux_label_dim > 0:
            obs_dict[self._aux_label_key] = gym.spaces.Box(
                -np.inf, np.inf, shape=(self._aux_label_dim,), dtype=np.float32
            )
        obs_space = gym.spaces.Dict(obs_dict)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        params["config"]["env_info"] = {
            "observation_space": obs_space,
            "action_space": action_space,
            "agents": 1,
            "value_size": 1,
        }
        params["config"]["num_actors"] = 1
        params["config"]["device"] = device
        params["config"]["device_name"] = device
        params["config"].setdefault("player", {})
        params["config"]["player"]["deterministic"] = bool(deterministic)
        params["config"]["player"]["use_vecenv"] = False

        self._torch = torch
        self._deterministic = bool(deterministic)
        self._runner = Runner()
        self._runner.load(cfg)
        self._player = self._runner.create_player()
        self._player.restore(checkpoint)
        self._initialized_batch = False

    @staticmethod
    def _detect_aux_label(params: dict) -> tuple[str, int]:
        """Return (label_key, dim) for a privileged aux-label obs group, or ("", 0) if none.

        Active only when the network has an ``aux_head`` whose ``label_key`` is also listed in
        ``env.obs_groups.obs`` (the explicit-estimator config). The label dimension is the widest slice
        the aux heads reference (w2's ``hole`` target is slice [1, 4] -> dim 4, matching the env's
        4-D label [grasp_angle, hole_gap_xyz] and the saved input-RMS aux_label shape).
        """
        aux_head = (params.get("network") or {}).get("aux_head") or {}
        label_key = aux_head.get("label_key")
        obs_list = ((params.get("env") or {}).get("obs_groups") or {}).get("obs") or []
        if not label_key or label_key not in obs_list:
            return "", 0
        targets = aux_head.get("targets") or {}
        dim = max((int(spec["slice"][1]) for spec in targets.values()), default=0)
        return str(label_key), int(dim)

    def reset(self) -> None:
        # Zero the recurrent (LSTM) hidden state so a new insertion episode starts fresh, matching
        # sim's per-episode reset. PpoPlayerContinuous.reset() -> init_rnn() needs self.batch_size,
        # which is only set by the first act(); guard so a pre-act reset just forces a clean re-init
        # on the next act() instead of raising.
        if self._initialized_batch and getattr(self._player, "is_rnn", False):
            self._player.reset()
        self._initialized_batch = False

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        # Explicit-estimator checkpoints expect the privileged `aux_label` key in the obs dict (the
        # saved input-RMS has a sub-module for it). It is excluded from the policy input and the aux
        # loss is skipped at inference, so dummy zeros have zero effect on the action; they only keep
        # the obs dict shape identical to training. Batch dim follows `policy` (deploy runs 1 env).
        if self._aux_label_dim > 0 and self._aux_label_key not in obs:
            batch = int(np.asarray(obs["policy"]).shape[0])
            obs = dict(obs)
            obs[self._aux_label_key] = np.zeros((batch, self._aux_label_dim), dtype=np.float32)
        obs_t = self._player.obs_to_torch(obs)
        if not self._initialized_batch:
            _ = self._player.get_batch_size(obs_t, 1)
            if self._player.is_rnn:
                self._player.init_rnn()
            self._initialized_batch = True
        with self._torch.no_grad():
            action = self._player.get_action(obs_t, is_deterministic=self._deterministic)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)

