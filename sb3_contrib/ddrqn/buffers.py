import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

class RecurrentReplayBuffer(ReplayBuffer):
    """
    Replay buffer for recurrent off-policy algorithms (DRQN/DDRQN).
    It samples sequence segments of length `sequence_length`.

    :param buffer_size: Max number of elements in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable memory optimization
    :param handle_timeout_termination: Handle timeout termination separately
    :param sequence_length: Number of consecutive steps in each sampled sequence
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
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.sequence_length = sequence_length
        # Array to store whether the transition is the start of an episode
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Track whether the previous step was done
        self._prev_dones = np.ones((self.n_envs,), dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._prev_dones = np.ones((self.n_envs,), dtype=np.float32)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: list) -> None:
        pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        # episode_starts is True if the previous transition was done
        self.episode_starts[pos] = self._prev_dones
        self._prev_dones = np.array(done, dtype=np.float32)

    def sample(self, batch_size: int, env = None) -> dict:
        """
        Sample a batch of sequence segments of length `self.sequence_length`.
        """
        max_idx = self.buffer_size if self.full else self.pos
        if max_idx <= self.sequence_length:
            raise ValueError(f"Not enough transitions in replay buffer to sample sequences of length {self.sequence_length}.")
            
        start_indices = []
        while len(start_indices) < batch_size:
            idx = np.random.randint(0, max_idx - self.sequence_length)
            # If buffer is full, ensure the sequence does not cross self.pos
            if self.full:
                seq_indices = np.arange(idx, idx + self.sequence_length) % self.buffer_size
                if self.pos in seq_indices:
                    continue
            start_indices.append(idx)
            
        return self._get_samples(start_indices)

    def _get_samples(self, start_indices: list[int]) -> dict:
        batch_size = len(start_indices)
        seq_len = self.sequence_length
        
        obs_seq = np.zeros((batch_size, seq_len, self.n_envs) + self.obs_shape, dtype=self.observations.dtype)
        next_obs_seq = np.zeros((batch_size, seq_len, self.n_envs) + self.obs_shape, dtype=self.observations.dtype)
        actions_seq = np.zeros((batch_size, seq_len, self.n_envs, self.action_dim), dtype=self.actions.dtype)
        rewards_seq = np.zeros((batch_size, seq_len, self.n_envs), dtype=np.float32)
        dones_seq = np.zeros((batch_size, seq_len, self.n_envs), dtype=np.float32)
        episode_starts_seq = np.zeros((batch_size, seq_len, self.n_envs), dtype=np.float32)
        
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
            
        # Reshape to flatten batch and env dimensions: (batch_size * n_envs, seq_len, ...)
        obs_seq = obs_seq.transpose(0, 2, 1, *range(3, len(obs_seq.shape))).reshape(batch_size * self.n_envs, seq_len, *self.obs_shape)
        next_obs_seq = next_obs_seq.transpose(0, 2, 1, *range(3, len(next_obs_seq.shape))).reshape(batch_size * self.n_envs, seq_len, *self.obs_shape)
        actions_seq = actions_seq.transpose(0, 2, 1, 3).reshape(batch_size * self.n_envs, seq_len, self.action_dim)
        rewards_seq = rewards_seq.transpose(0, 2, 1).reshape(batch_size * self.n_envs, seq_len)
        dones_seq = dones_seq.transpose(0, 2, 1).reshape(batch_size * self.n_envs, seq_len)
        episode_starts_seq = episode_starts_seq.transpose(0, 2, 1).reshape(batch_size * self.n_envs, seq_len)
        
        return dict(
            observations=self.to_torch(obs_seq),
            next_observations=self.to_torch(next_obs_seq),
            actions=self.to_torch(actions_seq),
            rewards=self.to_torch(rewards_seq).unsqueeze(-1),
            dones=self.to_torch(dones_seq).unsqueeze(-1),
            episode_starts=self.to_torch(episode_starts_seq).unsqueeze(-1)
        )
