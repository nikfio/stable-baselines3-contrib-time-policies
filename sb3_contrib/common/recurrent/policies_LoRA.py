import math
from typing import Any, Dict, Optional, Type
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.recurrent.policies import (
    RecurrentActorCriticPolicy,
    TimeCNN,
)


class LoraWeightParametrization(nn.Module):

    """
    LoRA (Low-Rank Adaptation) weight parametrization module.
    Computes: W_new = W_0 + (lora_B @ lora_A) * scaling
    """

    def __init__(self, original_shape: tuple, rank: int, scaling: float):
        super().__init__()
        out_features, in_features = original_shape
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = scaling

        # Initialize lora_A and lora_B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X + (self.lora_B @ self.lora_A) * self.scaling


class TimeCnnLstmPolicy(RecurrentActorCriticPolicy):
    """
    Recurrent policy that combines a 1D CNN feature extractor with
    an LSTM policy that can optionally be wrapped with LoRA.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param use_lora: Whether to apply LoRA (True) or standard training (False)
    :param lora_rank: Rank for the LoRA adapter matrices
    :param lora_alpha: Scaling factor (alpha) for the LoRA adapters
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        features_extractor_class: Type[BaseFeaturesExtractor] = TimeCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Call parent constructor (creates LSTM layers and initial optimizer)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )

        if self.use_lora:
            # Inject LoRA on Actor LSTM
            self._apply_lora(self.lstm_actor)

            # Inject LoRA on Critic LSTM (if separate critic LSTM is enabled)
            if self.lstm_critic is not None:
                self._apply_lora(self.lstm_critic)

            # Re-initialize the optimizer to only track parameters
            # with requires_grad=True.
            # This prevents AdamW from allocating VRAM for frozen base weights.
            self.optimizer = self.optimizer_class(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

    def _apply_lora(self, lstm_module: nn.LSTM) -> None:
        scaling = self.lora_alpha / self.lora_rank

        # Collect parameter names first to avoid dictionary changed size error
        names_to_parametrize = []
        for name, param in lstm_module.named_parameters():
            if "weight_ih_l" in name or "weight_hh_l" in name:
                names_to_parametrize.append(name)

        for name in names_to_parametrize:
            param = getattr(lstm_module, name)
            # Freeze original base weight parameter
            param.requires_grad = False
            # Register parametrization
            parametrize.register_parametrization(
                lstm_module,
                name,
                LoraWeightParametrization(
                    param.shape, self.lora_rank, scaling
                ),
            )
