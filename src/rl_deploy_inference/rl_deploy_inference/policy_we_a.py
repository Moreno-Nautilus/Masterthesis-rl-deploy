"""rl_games actor adapter for we_A_resnet18_baseline."""

from __future__ import annotations

import torch.nn as nn

from .policy import PolicyLoadError, RlGamesActor


def _patch_we_a_dict_rms() -> None:
    """Mirror the training-time normalizer: normalize vectors and pass images through."""
    import torch
    from rl_games.algos_torch import central_value, models
    from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs

    if getattr(torch.jit, "_we_a_rmsobs_patch", False):
        return

    class ImageSkipRMSObs(RunningMeanStdObs):
        def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
            nn.Module.__init__(self)
            self._passthrough = {key for key, shape in insize.items() if len(shape) != 1}
            self.running_mean_std = nn.ModuleDict(
                {
                    key: RunningMeanStd(shape, epsilon, per_channel, norm_only)
                    for key, shape in insize.items()
                    if len(shape) == 1
                }
            )

        def forward(self, input, denorm: bool = False):
            return {
                key: (value if key in self._passthrough else self.running_mean_std[key](value, denorm))
                for key, value in input.items()
            }

    original_script = torch.jit.script

    def _jit_script_skip_rmsobs(obj, *args, **kwargs):
        if isinstance(obj, RunningMeanStdObs):
            return obj
        return original_script(obj, *args, **kwargs)

    models.RunningMeanStdObs = ImageSkipRMSObs
    central_value.RunningMeanStdObs = ImageSkipRMSObs
    torch.jit.script = _jit_script_skip_rmsobs
    torch.jit._insertion_rmsobs_patch = True
    torch.jit._we_a_rmsobs_patch = True


class RlGamesActorWeA(RlGamesActor):
    """Fixed-shape actor for the saved we_A training contract."""

    def __init__(
        self,
        checkpoint: str,
        agent_config: str,
        train_repo: str,
        device: str,
        deterministic: bool,
    ):
        _patch_we_a_dict_rms()
        super().__init__(
            checkpoint=checkpoint,
            agent_config=agent_config,
            train_repo=train_repo,
            device=device,
            deterministic=deterministic,
            policy_dim=15,
            image_shape=(180, 320, 3),
            action_dim=6,
        )


__all__ = ["PolicyLoadError", "RlGamesActorWeA"]
