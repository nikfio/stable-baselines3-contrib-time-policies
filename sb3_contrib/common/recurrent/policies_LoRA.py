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


class LoraConv2dParametrization(nn.Module):
    """
    LoRA (Low-Rank Adaptation) weight parametrization module
    for 2D convolutions.
    Reshapes weight tensor from (out_channels, in_channels, kernel_height, kernel_width)
    to (out_channels, in_channels * kernel_height * kernel_width), applies standard LoRA,
    and reshapes back.
    """

    def __init__(self, original_shape: tuple, rank: int, scaling: float):
        super().__init__()
        self.original_shape = original_shape
        out_channels, num_features, k_h, k_w = original_shape

        self.lora_A = nn.Parameter(
            torch.zeros(rank, num_features * k_h * k_w)
        )
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        self.scaling = scaling

        # Initialize lora_A and lora_B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        return X + delta_W.view(self.original_shape)


class LoraConv1dParametrization(nn.Module):
    """
    LoRA (Low-Rank Adaptation) weight parametrization module
    for 1D convolutions.
    Reshapes weight tensor from (out_channels, in_channels, kernel_size)
    to (out_channels, in_channels * kernel_size), applies standard LoRA,
    and reshapes back.
    """

    def __init__(self, original_shape: tuple, rank: int, scaling: float):
        super().__init__()
        self.original_shape = original_shape
        out_channels, num_features, kernel_size = original_shape

        self.lora_A = nn.Parameter(
            torch.zeros(rank, num_features * kernel_size)
        )
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        self.scaling = scaling

        # Initialize lora_A and lora_B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        delta_W_2d = (self.lora_B @ self.lora_A) * self.scaling
        return X + delta_W_2d.view(self.original_shape)



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
    :param target_lstm: Whether to apply LoRA to the LSTM weight parameters
    :param target_cnn: Whether to apply LoRA to the CNN feature extractor
    :param target_mlp: Whether to apply LoRA to the MLP policy/value
        projection heads
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        target_lstm: bool = True,
        target_cnn: bool = True,
        target_mlp: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = TimeCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_lstm = target_lstm
        self.target_cnn = target_cnn
        self.target_mlp = target_mlp

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
            # 1. Apply LoRA to LSTM layers
            if self.target_lstm:
                self._apply_lora_to_lstm(self.lstm_actor)
                if self.lstm_critic is not None:
                    self._apply_lora_to_lstm(self.lstm_critic)

            # 2. Apply LoRA to CNN Feature Extractor
            if self.target_cnn:
                self._apply_lora_to_module(self.features_extractor)

            # 3. Apply LoRA to MLP heads (mlp_extractor, action_net, value_net)
            if self.target_mlp:
                self._apply_lora_to_module(self.mlp_extractor)
                self._apply_lora_to_module(self.action_net)
                self._apply_lora_to_module(self.value_net)

            # Re-initialize the optimizer to only track parameters
            # with requires_grad=True.
            self.optimizer = self.optimizer_class(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

    def _process_sequence(
        self,
        features: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        lstm: nn.LSTM,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Override parent _process_sequence to wrap LSTM calls inside
        ``parametrize.cached()`` when LoRA is active.

        Without caching, every ``lstm.forward()`` triggers
        ``_update_flat_weights → _weights_have_changed → getattr``
        which re-evaluates the LoRA parametrization and materializes
        a **new** full-size weight tensor on GPU each time.
        In the per-step loop (128 steps × 16 weight matrices × 2
        LSTMs × 5 PPO epochs) this creates ~15 GB of transient
        tensors in the autograd graph, causing OOM on 20 GB cards.

        ``parametrize.cached()`` computes each parametrized weight
        once and reuses it for all accesses within the context.
        """
        if self.use_lora and self.target_lstm:
            with parametrize.cached():
                return super()._process_sequence(
                    features, lstm_states, episode_starts, lstm
                )
        return super()._process_sequence(
            features, lstm_states, episode_starts, lstm
        )

    def _apply_lora_to_lstm(self, lstm_module: nn.LSTM) -> None:
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

    def _apply_lora_to_module(self, module: nn.Module) -> None:
        scaling = self.lora_alpha / self.lora_rank

        # Traverse all child/sub-modules recursively
        for _, submodule in module.named_modules():
            # If the submodule is nn.Linear, apply standard 2D LoRA
            if isinstance(submodule, nn.Linear):
                submodule.weight.requires_grad = False
                parametrize.register_parametrization(
                    submodule,
                    "weight",
                    LoraWeightParametrization(
                        submodule.weight.shape, self.lora_rank, scaling
                    ),
                )
            # If the submodule is nn.Conv1d, apply 1D convolutional LoRA
            elif isinstance(submodule, nn.Conv1d):
                submodule.weight.requires_grad = False
                parametrize.register_parametrization(
                    submodule,
                    "weight",
                    LoraConv1dParametrization(
                        submodule.weight.shape, self.lora_rank, scaling
                    ),
                )
            # If the submodule is nn.Conv2d, apply 2D convolutional LoRA
            elif isinstance(submodule, nn.Conv2d):
                submodule.weight.requires_grad = False
                parametrize.register_parametrization(
                    submodule,
                    "weight",
                    LoraConv2dParametrization(
                        submodule.weight.shape, self.lora_rank, scaling
                    ),
                )
