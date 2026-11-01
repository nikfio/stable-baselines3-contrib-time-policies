from typing import Any
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape


class RecurrentReplayBuffer(ReplayBuffer):
    """
    Replay buffer for recurrent off-policy algorithms (DRQN/DDRQN).
    It samples sequence segments of length `sequence_length`.
    Supports both `spaces.Box` and `spaces.Dict` observations, optional
    `float16` storage compression for 3D market tensors, and `action_masks`
    storage for masked Double-DQN target calculation.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        sequence_length: int = 16,
        storage_dtype: str = "float32",
    ):
        self.is_dict_obs = isinstance(observation_space, spaces.Dict)
        self.storage_dtype_str = str(storage_dtype).lower()
        self.market_np_dtype = (
            np.float16
            if self.storage_dtype_str in ("float16", "fp16")
            else np.float32
        )

        # Pass a tiny 1-byte dummy Box to super().__init__ when using Dict or float16
        # to avoid allocating a redundant multi-GB float32 buffer during parent init.
        dummy_space = (
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8)
            if (self.is_dict_obs or self.market_np_dtype == np.float16)
            else observation_space
        )
        super().__init__(
            buffer_size,
            dummy_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.observation_space = observation_space
        self.obs_shape = get_obs_shape(observation_space)
        self.sequence_length = sequence_length
        self.n_actions = int(action_space.n) if isinstance(action_space, spaces.Discrete) else 1

        if self.is_dict_obs:
            assert isinstance(self.obs_shape, dict)
            self.observations: Any = {
                key: np.zeros(
                    (self.buffer_size, self.n_envs, *shape),
                    dtype=(self.market_np_dtype if len(shape) >= 3 else np.float32),
                )
                for key, shape in self.obs_shape.items()
            }
            if self.optimize_memory_usage:
                self.next_observations: Any = None
            else:
                self.next_observations = {
                    key: np.zeros(
                        (self.buffer_size, self.n_envs, *shape),
                        dtype=(self.market_np_dtype if len(shape) >= 3 else np.float32),
                    )
                    for key, shape in self.obs_shape.items()
                }
        elif self.market_np_dtype == np.float16:
            assert isinstance(self.obs_shape, tuple)
            self.observations = np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape),
                dtype=np.float16,
            )
            if self.optimize_memory_usage:
                self.next_observations = None
            else:
                self.next_observations = np.zeros(
                    (self.buffer_size, self.n_envs, *self.obs_shape),
                    dtype=np.float16,
                )

        # Array to store whether the transition is the start of an episode
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._prev_dones = np.ones((self.n_envs,), dtype=np.float32)
        # Action masks for next_observations (used in Double-DQN argmax target)
        self.next_action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.n_actions), dtype=bool
        )

    def reset(self) -> None:
        super().reset()
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._prev_dones = np.ones((self.n_envs,), dtype=np.float32)
        self.next_action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.n_actions), dtype=bool
        )

    def add(
        self,
        obs: np.ndarray | dict[str, np.ndarray],
        next_obs: np.ndarray | dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        pos = self.pos
        if self.is_dict_obs:
            assert isinstance(obs, dict) and isinstance(next_obs, dict)
            for key in self.observations:
                target_dtype = self.observations[key].dtype
                self.observations[key][pos] = np.asarray(obs[key], dtype=target_dtype)
                if self.optimize_memory_usage:
                    self.observations[key][(pos + 1) % self.buffer_size] = np.asarray(
                        next_obs[key], dtype=target_dtype
                    )
                else:
                    self.next_observations[key][pos] = np.asarray(
                        next_obs[key], dtype=target_dtype
                    )
            action = action.reshape((self.n_envs, self.action_dim))
            self.actions[pos] = np.array(action)
            self.rewards[pos] = np.array(reward)
            self.dones[pos] = np.array(done)
            if self.handle_timeout_termination:
                self.timeouts[pos] = np.array(
                    [info.get("TimeLimit.truncated", False) for info in infos]
                )
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
        else:
            super().add(obs, next_obs, action, reward, done, infos)

        self.episode_starts[pos] = self._prev_dones
        self._prev_dones = np.array(done, dtype=np.float32)

        if infos:
            for env_idx, info in enumerate(infos):
                if isinstance(info, dict) and "action_mask" in info:
                    mask = np.asarray(info["action_mask"], dtype=bool).reshape(-1)
                    if mask.shape[0] == self.n_actions:
                        self.next_action_masks[pos, env_idx] = mask
                    else:
                        self.next_action_masks[pos, env_idx] = True
                else:
                    self.next_action_masks[pos, env_idx] = True

    def sample(self, batch_size: int, env: Any = None) -> dict[str, Any]:
        """
        Sample a batch of sequence segments of length `self.sequence_length`.
        """
        max_idx = self.buffer_size if self.full else self.pos
        if max_idx <= self.sequence_length:
            raise ValueError(
                f"Not enough transitions in replay buffer to sample sequences of length {self.sequence_length}."
            )

        start_indices: list[int] = []
        while len(start_indices) < batch_size:
            idx = int(np.random.randint(0, max_idx - self.sequence_length))
            if self.full:
                seq_indices = np.arange(idx, idx + self.sequence_length) % self.buffer_size
                if self.pos in seq_indices:
                    continue
            start_indices.append(idx)

        return self._get_samples(start_indices)

    def _get_samples(self, start_indices: list[int]) -> dict[str, Any]:
        batch_size = len(start_indices)
        seq_len = self.sequence_length

        actions_seq = np.zeros(
            (batch_size, seq_len, self.n_envs, self.action_dim), dtype=self.actions.dtype
        )
        rewards_seq = np.zeros((batch_size, seq_len, self.n_envs), dtype=np.float32)
        dones_seq = np.zeros((batch_size, seq_len, self.n_envs), dtype=np.float32)
        episode_starts_seq = np.zeros((batch_size, seq_len, self.n_envs), dtype=np.float32)
        next_masks_seq = np.ones(
            (batch_size, seq_len, self.n_envs, self.n_actions), dtype=bool
        )

        if self.is_dict_obs:
            assert isinstance(self.obs_shape, dict)
            obs_seq_dict = {
                key: np.zeros(
                    (batch_size, seq_len, self.n_envs, *shape),
                    dtype=self.observations[key].dtype,
                )
                for key, shape in self.obs_shape.items()
            }
            next_obs_seq_dict = {
                key: np.zeros(
                    (batch_size, seq_len, self.n_envs, *shape),
                    dtype=self.observations[key].dtype,
                )
                for key, shape in self.obs_shape.items()
            }
            for i, start_idx in enumerate(start_indices):
                indices = np.arange(start_idx, start_idx + seq_len) % self.buffer_size
                for key in self.obs_shape:
                    obs_seq_dict[key][i] = self.observations[key][indices]
                    if self.optimize_memory_usage:
                        next_obs_seq_dict[key][i] = self.observations[key][
                            (indices + 1) % self.buffer_size
                        ]
                    else:
                        next_obs_seq_dict[key][i] = self.next_observations[key][indices]
                actions_seq[i] = self.actions[indices]
                rewards_seq[i] = self.rewards[indices]
                dones_seq[i] = self.dones[indices]
                episode_starts_seq[i] = self.episode_starts[indices]
                next_masks_seq[i] = self.next_action_masks[indices]

            obs_torch: Any = {}
            next_obs_torch: Any = {}
            for key, shape in self.obs_shape.items():
                o_arr = obs_seq_dict[key].transpose(
                    0, 2, 1, *range(3, len(obs_seq_dict[key].shape))
                ).reshape(batch_size * self.n_envs, seq_len, *shape)
                no_arr = next_obs_seq_dict[key].transpose(
                    0, 2, 1, *range(3, len(next_obs_seq_dict[key].shape))
                ).reshape(batch_size * self.n_envs, seq_len, *shape)
                obs_torch[key] = self.to_torch(o_arr).float()
                next_obs_torch[key] = self.to_torch(no_arr).float()
        else:
            assert isinstance(self.obs_shape, tuple)
            obs_seq = np.zeros(
                (batch_size, seq_len, self.n_envs) + self.obs_shape,
                dtype=self.observations.dtype,
            )
            next_obs_seq = np.zeros(
                (batch_size, seq_len, self.n_envs) + self.obs_shape,
                dtype=self.observations.dtype,
            )
            for i, start_idx in enumerate(start_indices):
                indices = np.arange(start_idx, start_idx + seq_len) % self.buffer_size
                obs_seq[i] = self.observations[indices]
                if self.optimize_memory_usage:
                    next_obs_seq[i] = self.observations[(indices + 1) % self.buffer_size]
                else:
                    next_obs_seq[i] = self.next_observations[indices]
                actions_seq[i] = self.actions[indices]
                rewards_seq[i] = self.rewards[indices]
                dones_seq[i] = self.dones[indices]
                episode_starts_seq[i] = self.episode_starts[indices]
                next_masks_seq[i] = self.next_action_masks[indices]

            obs_seq = obs_seq.transpose(
                0, 2, 1, *range(3, len(obs_seq.shape))
            ).reshape(batch_size * self.n_envs, seq_len, *self.obs_shape)
            next_obs_seq = next_obs_seq.transpose(
                0, 2, 1, *range(3, len(next_obs_seq.shape))
            ).reshape(batch_size * self.n_envs, seq_len, *self.obs_shape)
            obs_torch = self.to_torch(obs_seq).float()
            next_obs_torch = self.to_torch(next_obs_seq).float()

        actions_seq = actions_seq.transpose(0, 2, 1, 3).reshape(
            batch_size * self.n_envs, seq_len, self.action_dim
        )
        rewards_seq = rewards_seq.transpose(0, 2, 1).reshape(
            batch_size * self.n_envs, seq_len
        )
        dones_seq = dones_seq.transpose(0, 2, 1).reshape(
            batch_size * self.n_envs, seq_len
        )
        episode_starts_seq = episode_starts_seq.transpose(0, 2, 1).reshape(
            batch_size * self.n_envs, seq_len
        )
        next_masks_seq = next_masks_seq.transpose(0, 2, 1, 3).reshape(
            batch_size * self.n_envs, seq_len, self.n_actions
        )

        return dict(
            observations=obs_torch,
            next_observations=next_obs_torch,
            actions=self.to_torch(actions_seq),
            rewards=self.to_torch(rewards_seq).unsqueeze(-1),
            dones=self.to_torch(dones_seq).unsqueeze(-1),
            episode_starts=self.to_torch(episode_starts_seq).unsqueeze(-1),
            next_action_masks=self.to_torch(next_masks_seq).bool(),
        )
