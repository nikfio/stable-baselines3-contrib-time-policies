from typing import Any, Optional, Type
import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.recurrent.policies import TimeCNN

class DdrqnQNetwork(nn.Module):
    """
    Q-Network containing TimeCNN + LSTM + Dueling heads (Value & Advantage).
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
    ):
        super().__init__()
        self.action_space = action_space
        self.features_dim = features_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.features_extractor = features_extractor

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=features_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=False  # Keep false to align with process_sequence (sequence_length, batch_size, ...)
        )

        # Dueling Streams: Value & Advantage
        val_layers = []
        in_dim = lstm_hidden_size
        for h_dim in val_head_arch:
            val_layers.append(nn.Linear(in_dim, h_dim))
            val_layers.append(nn.ReLU())
            in_dim = h_dim
        val_layers.append(nn.Linear(in_dim, 1))
        self.value_head = nn.Sequential(*val_layers)

        adv_layers = []
        in_dim = lstm_hidden_size
        for h_dim in adv_head_arch:
            adv_layers.append(nn.Linear(in_dim, h_dim))
            adv_layers.append(nn.ReLU())
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
        Do a forward pass in the LSTM network, resetting states when episode_starts is 1.
        """
        n_seq = lstm_states[0].shape[1]
        
        # Batch to sequence: (batch_size * sequence_length, features_dim) -> (sequence_length, batch_size, features_dim)
        features_sequence = features.reshape((n_seq, -1, self.lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = self.lstm(features_sequence, lstm_states)
            lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        for features_step, episode_start in zip(features_sequence, episode_starts, strict=True):
            hidden, lstm_states = self.lstm(
                features_step.unsqueeze(dim=0),
                (
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
            
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        features = self.features_extractor(obs)
        lstm_out, lstm_states = self._process_sequence(features, lstm_states, episode_starts)
        
        value = self.value_head(lstm_out)
        advantages = self.advantage_head(lstm_out)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values, lstm_states

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class TimeCnnLstmDdrqnPolicy(BasePolicy):
    """
    TimeCnnLstmDdrqnPolicy is a recurrent DQN policy with Dueling heads and TimeCNN feature extractor.
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
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.val_head_arch = val_head_arch
        self.adv_head_arch = adv_head_arch
        
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
        ).to(self.device)

    def predict(
        self,
        observation: Any,
        state: Optional[tuple[np.ndarray, np.ndarray]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        self.set_training_mode(False)
        
        with th.no_grad():
            if isinstance(observation, th.Tensor):
                obs_tensor = observation.to(self.device)
                vectorized_env = True
            else:
                obs_tensor, vectorized_env = self.obs_to_tensor(observation)
            
            n_envs = obs_tensor.shape[0]
            
            if state is None:
                state_tensor = (
                    th.zeros(self.n_lstm_layers, n_envs, self.lstm_hidden_size, device=self.device),
                    th.zeros(self.n_lstm_layers, n_envs, self.lstm_hidden_size, device=self.device)
                )
            else:
                state_tensor = (
                    th.as_tensor(state[0], dtype=th.float32, device=self.device),
                    th.as_tensor(state[1], dtype=th.float32, device=self.device)
                )
            
            if episode_start is None:
                episode_start_tensor = th.zeros(n_envs, device=self.device)
            else:
                episode_start_tensor = th.as_tensor(episode_start, device=self.device).float()
                
            q_values, next_state_tensor = self.q_net(obs_tensor, state_tensor, episode_start_tensor)
            
            if not deterministic:
                probs = th.softmax(q_values / temperature, dim=-1)
                # Sample from the probability distribution
                actions_tensor = th.multinomial(probs, num_samples=1).squeeze(dim=-1)
            else:
                actions_tensor = q_values.argmax(dim=-1)
            
            actions = actions_tensor.cpu().numpy().reshape((-1, *self.action_space.shape))
            if not vectorized_env:
                actions = actions.squeeze(axis=0)
                
            next_state = (
                next_state_tensor[0].cpu().numpy(),
                next_state_tensor[1].cpu().numpy()
            )
            
        return actions, next_state

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = True,
    ) -> th.Tensor:
        n_envs = observation.shape[0]
        state = (
            th.zeros(self.n_lstm_layers, n_envs, self.lstm_hidden_size, device=self.device),
            th.zeros(self.n_lstm_layers, n_envs, self.lstm_hidden_size, device=self.device)
        )
        episode_start = th.zeros(n_envs, device=self.device)
        q_values, _ = self.q_net(observation, state, episode_start)
        return q_values.argmax(dim=-1)

    def set_training_mode(self, mode: bool) -> None:
        self.q_net.train(mode)
        self.training = mode
