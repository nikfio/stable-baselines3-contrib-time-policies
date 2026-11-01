from typing import Any, Optional, Type
import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.recurrent.policies import (
    TimeCNN,
    _resolve_activation_fn,
)


class DdrqnQNetwork(nn.Module):
    """
    Q-Network containing TimeCNN + LSTM + Dueling heads (Value & Advantage).
    Supports both Box observations and Dict({"market": ..., "portfolio": ...})
    observations with portfolio skip-connections into the LSTM and Dueling
    heads.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        val_head_arch: list[int] = [32],
        adv_head_arch: list[int] = [32],
        use_layer_norm: bool = False,
        activation_fn: Type[nn.Module] | str = nn.ReLU,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.use_layer_norm = bool(use_layer_norm)
        self.activation_fn = _resolve_activation_fn(activation_fn)
        self.features_extractor = features_extractor

        if (
            isinstance(observation_space, spaces.Dict)
            and "portfolio" in observation_space.spaces
        ):
            self.portfolio_dim = int(
                np.prod(observation_space.spaces["portfolio"].shape)
            )
        else:
            self.portfolio_dim = 0

        # LSTM Layer (takes CNN features + optional portfolio state)
        self.lstm = nn.LSTM(
            input_size=features_dim + self.portfolio_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=False,
        )

        # Dueling Streams: Value & Advantage (with optional portfolio skip)
        head_in_dim = lstm_hidden_size + self.portfolio_dim

        val_layers: list[nn.Module] = []
        in_dim = head_in_dim
        for h_dim in val_head_arch:
            val_layers.append(nn.Linear(in_dim, h_dim))
            if self.use_layer_norm:
                val_layers.append(nn.LayerNorm(h_dim))
            val_layers.append(self.activation_fn())
            in_dim = h_dim
        val_layers.append(nn.Linear(in_dim, 1))
        self.value_head = nn.Sequential(*val_layers)

        adv_layers: list[nn.Module] = []
        in_dim = head_in_dim
        for h_dim in adv_head_arch:
            adv_layers.append(nn.Linear(in_dim, h_dim))
            if self.use_layer_norm:
                adv_layers.append(nn.LayerNorm(h_dim))
            adv_layers.append(self.activation_fn())
            in_dim = h_dim
        adv_layers.append(nn.Linear(in_dim, action_space.n))
        self.advantage_head = nn.Sequential(*adv_layers)

    def _process_sequence(
        self,
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Do a forward pass in the LSTM network, resetting states when
        episode_starts is 1.
        """
        n_seq = lstm_states[0].shape[1]

        # Batch to sequence:
        # (batch_size * seq_len, features_dim) -> (seq_len, batch_size, dim)
        features_sequence = features.reshape(
            (n_seq, -1, self.lstm.input_size)
        ).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = self.lstm(
                features_sequence, lstm_states
            )
            lstm_output = th.flatten(
                lstm_output.transpose(0, 1), start_dim=0, end_dim=1
            )
            return lstm_output, lstm_states

        lstm_output = []
        for features_step, episode_start in zip(
            features_sequence, episode_starts, strict=True
        ):
            hidden, lstm_states = self.lstm(
                features_step.unsqueeze(dim=0),
                (
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]

        lstm_output = th.flatten(
            th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1
        )
        return lstm_output, lstm_states

    def _extract_features_and_portfolio(
        self, obs: th.Tensor | dict[str, th.Tensor]
    ) -> tuple[th.Tensor, Optional[th.Tensor]]:
        features = self.features_extractor(obs)
        if self.portfolio_dim <= 0:
            return features, None
        if isinstance(obs, dict) and "portfolio" in obs:
            portfolio = obs["portfolio"].float().view(
                features.shape[0], self.portfolio_dim
            )
        else:
            portfolio = th.zeros(
                (features.shape[0], self.portfolio_dim),
                dtype=features.dtype,
                device=features.device,
            )
        return features, portfolio

    def forward(
        self,
        obs: th.Tensor | dict[str, th.Tensor],
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        features, portfolio = self._extract_features_and_portfolio(obs)
        if portfolio is not None:
            lstm_input = th.cat([features, portfolio], dim=-1)
        else:
            lstm_input = features

        lstm_out, lstm_states = self._process_sequence(
            lstm_input, lstm_states, episode_starts
        )

        if portfolio is not None:
            head_input = th.cat([lstm_out, portfolio], dim=-1)
        else:
            head_input = lstm_out

        value = self.value_head(head_input)
        advantages = self.advantage_head(head_input)
        q_values = value + (
            advantages - advantages.mean(dim=-1, keepdim=True)
        )

        return q_values, lstm_states

    def forward_actor_and_value(
        self, obs: th.Tensor | dict[str, th.Tensor]
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Stateless single-step forward helper for model decomposition /
        functional sensitivity probing. Returns (q_values, value).
        """
        features, portfolio = self._extract_features_and_portfolio(obs)
        batch_size = features.shape[0]
        device = features.device
        lstm_states = (
            th.zeros(
                self.n_lstm_layers,
                batch_size,
                self.lstm_hidden_size,
                device=device,
            ),
            th.zeros(
                self.n_lstm_layers,
                batch_size,
                self.lstm_hidden_size,
                device=device,
            ),
        )
        episode_starts = th.zeros(batch_size, device=device)
        if portfolio is not None:
            lstm_input = th.cat([features, portfolio], dim=-1)
        else:
            lstm_input = features

        lstm_out, _ = self._process_sequence(
            lstm_input, lstm_states, episode_starts
        )
        if portfolio is not None:
            head_input = th.cat([lstm_out, portfolio], dim=-1)
        else:
            head_input = lstm_out

        value = self.value_head(head_input)
        advantages = self.advantage_head(head_input)
        q_values = value + (
            advantages - advantages.mean(dim=-1, keepdim=True)
        )
        return q_values, value

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class TimeCnnLstmDdrqnPolicy(BasePolicy):
    """
    TimeCnnLstmDdrqnPolicy is a recurrent DQN policy with Dueling heads
    and TimeCNN feature extractor.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = TimeCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        val_head_arch: list[int] = [32],
        adv_head_arch: list[int] = [32],
        use_layer_norm: bool = False,
        activation_fn: Type[nn.Module] | str = nn.ReLU,
    ):
        extractor_kwargs = dict(features_extractor_kwargs or {})
        if features_extractor_class is TimeCNN:
            extractor_kwargs.setdefault("use_layer_norm", use_layer_norm)
            extractor_kwargs.setdefault("activation_fn", activation_fn)

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.val_head_arch = val_head_arch
        self.adv_head_arch = adv_head_arch
        self.use_layer_norm = bool(use_layer_norm)
        self.activation_fn = _resolve_activation_fn(activation_fn)

        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        self.optimizer = self.optimizer_class(
            self.q_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> DdrqnQNetwork:
        features_extractor = self.make_features_extractor()
        return DdrqnQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=features_extractor,
            features_dim=self.features_dim,
            lstm_hidden_size=self.lstm_hidden_size,
            n_lstm_layers=self.n_lstm_layers,
            val_head_arch=self.val_head_arch,
            adv_head_arch=self.adv_head_arch,
            use_layer_norm=self.use_layer_norm,
            activation_fn=self.activation_fn,
        ).to(self.device)

    @staticmethod
    def _batch_size_from_obs_tensor(
        obs_tensor: th.Tensor | dict[str, th.Tensor]
    ) -> int:
        if isinstance(obs_tensor, dict):
            first_tensor = next(iter(obs_tensor.values()))
            return int(first_tensor.shape[0])
        return int(obs_tensor.shape[0])

    def _apply_action_masks(
        self,
        q_values: th.Tensor,
        action_masks: Optional[np.ndarray | th.Tensor],
    ) -> tuple[th.Tensor, Optional[th.Tensor]]:
        if action_masks is None:
            return q_values, None
        mask_tensor = th.as_tensor(
            action_masks, dtype=th.bool, device=q_values.device
        ).view(-1, self.action_space.n)
        # Guard against an all-False row
        valid_rows = mask_tensor.any(dim=-1, keepdim=True)
        safe_mask = th.where(
            valid_rows, mask_tensor, th.ones_like(mask_tensor)
        )
        masked_q = q_values.masked_fill(~safe_mask, -1e8)
        return masked_q, safe_mask

    def predict(
        self,
        observation: Any,
        state: Optional[tuple[np.ndarray, np.ndarray]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
        temperature: float = 1.0,
        action_masks: Optional[np.ndarray | th.Tensor] = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        self.set_training_mode(False)

        with th.no_grad():
            if isinstance(observation, th.Tensor):
                obs_tensor: th.Tensor | dict[str, th.Tensor] = observation.to(
                    self.device
                )
                vectorized_env = True
            elif isinstance(observation, dict) and all(
                isinstance(v, th.Tensor) for v in observation.values()
            ):
                obs_tensor = {
                    k: v.to(self.device) for k, v in observation.items()
                }
                vectorized_env = True
            else:
                obs_tensor, vectorized_env = self.obs_to_tensor(observation)

            n_envs = self._batch_size_from_obs_tensor(obs_tensor)

            if state is None:
                state_tensor = (
                    th.zeros(
                        self.n_lstm_layers,
                        n_envs,
                        self.lstm_hidden_size,
                        device=self.device,
                    ),
                    th.zeros(
                        self.n_lstm_layers,
                        n_envs,
                        self.lstm_hidden_size,
                        device=self.device,
                    ),
                )
            else:
                state_tensor = (
                    th.as_tensor(
                        state[0], dtype=th.float32, device=self.device
                    ),
                    th.as_tensor(
                        state[1], dtype=th.float32, device=self.device
                    ),
                )

            if episode_start is None:
                episode_start_tensor = th.zeros(n_envs, device=self.device)
            else:
                episode_start_tensor = th.as_tensor(
                    episode_start, device=self.device
                ).float()

            q_values, next_state_tensor = self.q_net(
                obs_tensor, state_tensor, episode_start_tensor
            )

            safe_q = (
                q_values
                if th.isfinite(q_values).all()
                else th.nan_to_num(
                    q_values, nan=0.0, posinf=50.0, neginf=-50.0
                )
            )
            masked_q, safe_mask = self._apply_action_masks(
                safe_q, action_masks
            )

            if not deterministic:
                temp = max(float(temperature), 1e-4)
                logits = masked_q.float() / temp
                if safe_mask is not None:
                    logits = logits.masked_fill(~safe_mask, -1e8)
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs = th.softmax(logits, dim=-1)
                if safe_mask is not None:
                    probs = probs * safe_mask.float()
                    row_sums = probs.sum(dim=-1, keepdim=True)
                    probs = th.where(
                        row_sums > 0,
                        probs / row_sums.clamp(min=1e-8),
                        safe_mask.float()
                        / safe_mask.float().sum(dim=-1, keepdim=True).clamp(
                            min=1.0
                        ),
                    )
                elif (
                    not th.isfinite(probs).all()
                    or not (probs.sum(dim=-1) > 0).all()
                ):
                    probs = th.full_like(
                        probs, 1.0 / float(self.action_space.n)
                    )
                actions_tensor = th.multinomial(
                    probs, num_samples=1
                ).squeeze(dim=-1)
            else:
                actions_tensor = masked_q.argmax(dim=-1)

            if (
                not th.isfinite(next_state_tensor[0]).all()
                or not th.isfinite(next_state_tensor[1]).all()
            ):
                next_state_tensor = (
                    th.nan_to_num(
                        next_state_tensor[0], nan=0.0, posinf=1.0, neginf=-1.0
                    ),
                    th.nan_to_num(
                        next_state_tensor[1], nan=0.0, posinf=1.0, neginf=-1.0
                    ),
                )

            actions = actions_tensor.cpu().numpy().reshape(
                (-1, *self.action_space.shape)
            )
            if not vectorized_env:
                actions = actions.squeeze(axis=0)

            next_state = (
                next_state_tensor[0].cpu().numpy(),
                next_state_tensor[1].cpu().numpy(),
            )

        return actions, next_state

    def _predict(
        self,
        observation: th.Tensor | dict[str, th.Tensor],
        deterministic: bool = True,
    ) -> th.Tensor:
        n_envs = self._batch_size_from_obs_tensor(observation)
        state = (
            th.zeros(
                self.n_lstm_layers,
                n_envs,
                self.lstm_hidden_size,
                device=self.device,
            ),
            th.zeros(
                self.n_lstm_layers,
                n_envs,
                self.lstm_hidden_size,
                device=self.device,
            ),
        )
        episode_start = th.zeros(n_envs, device=self.device)
        q_values, _ = self.q_net(observation, state, episode_start)
        return q_values.argmax(dim=-1)

    def set_training_mode(self, mode: bool) -> None:
        self.q_net.train(mode)
        self.training = mode
