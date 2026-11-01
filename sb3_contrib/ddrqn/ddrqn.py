import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
)
from stable_baselines3.common.utils import (
    LinearSchedule,
    get_parameters_by_name,
    polyak_update,
    should_collect_more_steps,
)
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.ddrqn.buffers import RecurrentReplayBuffer
from sb3_contrib.ddrqn.policies import TimeCnnLstmDdrqnPolicy

SelfDDRQN = TypeVar("SelfDDRQN", bound="DDRQN")


class DDRQN(OffPolicyAlgorithm):
    """
    Dueling Deep Recurrent Q-Network (DDRQN)
    Specifically tailored for recurrent sequence learning over time-series
    inputs.

    :param policy: The policy model to use (TimeCnnLstmDdrqnPolicy, ...)
    :param env: The environment to learn from
    :param learning_rate: The learning rate (float or Schedule)
    :param buffer_size: Size of the replay buffer (number of transitions)
    :param learning_starts: How many steps of the model to collect transitions
        for before learning starts
    :param batch_size: Minibatch size (number of sequences) for each update
    :param tau: The soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: The discount factor
    :param train_freq: Update the model every ``train_freq`` steps
    :param gradient_steps: How many gradient steps to do after each rollout
    :param target_update_interval: Update target network every
        ``target_update_interval`` env steps
    :param exploration_fraction: Fraction of entire training period over which
        exploration rate decays
    :param exploration_initial_eps: Initial value of random action probability
    :param exploration_final_eps: Final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param sequence_length: Sequence length for BPTT
    :param burn_in_steps: Number of initial steps in sampled sequences to warm
        up recurrent hidden states
    :param stats_window_size: Window size for the rollout logging
    :param tensorboard_log: The log location for tensorboard
    :param policy_kwargs: Additional arguments to be passed to the policy
    :param verbose: Verbosity level
    :param seed: Seed for the pseudo random generators
    :param device: Device on which the code should be run
    :param _init_setup_model: Whether or not to build the network at creation
    """
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "TimeCnnLstmDdrqnPolicy": TimeCnnLstmDdrqnPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[TimeCnnLstmDdrqnPolicy]],
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 50000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.02,
        exploration_strategy: str = "epsilon-greedy",
        exploration_initial_temp: float = 1.0,
        exploration_final_temp: float = 0.1,
        max_grad_norm: float = 1.0,
        sequence_length: int = 16,
        burn_in_steps: int = 4,
        n_steps: int = 1,
        replay_buffer_storage_dtype: str = "float32",
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.n_steps = max(1, int(n_steps))
        self.replay_buffer_storage_dtype = str(replay_buffer_storage_dtype)
        # We need a custom replay buffer that returns sequences with enough
        # lookahead steps to compute exact n-step TD targets
        replay_buffer_kwargs = {
            "sequence_length": (
                sequence_length + burn_in_steps + max(0, self.n_steps - 1)
            ),
            "storage_dtype": self.replay_buffer_storage_dtype,
        }

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,
            replay_buffer_class=RecurrentReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_strategy = exploration_strategy
        self.exploration_initial_temp = exploration_initial_temp
        self.exploration_final_temp = exploration_final_temp
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.sequence_length = sequence_length
        self.burn_in_steps = burn_in_steps

        self._n_calls = 0
        self.exploration_rate = float(self.exploration_initial_eps)
        self.exploration_temp = float(self.exploration_initial_temp)
        self._last_lstm_states = None
        self._last_action_masks: Optional[np.ndarray] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats for batch norm (if any)
        self.batch_norm_stats = get_parameters_by_name(
            self.q_net, ["running_"]
        )
        self.batch_norm_stats_target = get_parameters_by_name(
            self.q_net_target, ["running_"]
        )

        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        self.temp_schedule = LinearSchedule(
            self.exploration_initial_temp,
            self.exploration_final_temp,
            self.exploration_fraction,
        )
        self.exploration_rate = float(
            self.exploration_schedule(self._current_progress_remaining)
        )
        self.exploration_temp = float(
            self.temp_schedule(self._current_progress_remaining)
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        Called in ``collect_rollouts()`` after each step.
        """
        self._n_calls += 1
        # Sync target network
        interval = max(self.target_update_interval // self.n_envs, 1)
        if self._n_calls % interval == 0:
            polyak_update(
                self.q_net.parameters(),
                self.q_net_target.parameters(),
                self.tau,
            )
            # Polyak updates of batch norm stats
            polyak_update(
                self.batch_norm_stats, self.batch_norm_stats_target, 1.0
            )

        self.exploration_rate = self.exploration_schedule(
            self._current_progress_remaining
        )
        self.exploration_temp = self.temp_schedule(
            self._current_progress_remaining
        )
        self.logger.record("rollout/exploration_rate", self.exploration_rate)
        self.logger.record("rollout/exploration_temp", self.exploration_temp)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: RecurrentReplayBuffer,
        action_noise: Optional[Any] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Custom collect_rollouts to track and update the recurrent LSTM states.
        """
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        callback.on_rollout_start()

        # Initialize or maintain LSTM states
        if self._last_lstm_states is None:
            shape = (
                self.policy.n_lstm_layers,
                env.num_envs,
                self.policy.lstm_hidden_size,
            )
            self._last_lstm_states = (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
            )

        if self._last_action_masks is None:
            try:
                masks = env.env_method("action_masks")
                if masks is not None and all(m is not None for m in masks):
                    self._last_action_masks = np.asarray(
                        masks, dtype=np.bool_
                    ).reshape(env.num_envs, self.action_space.n)
            except Exception:
                self._last_action_masks = None

        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if self.num_timesteps < learning_starts:
                actions = self._sample_random_actions(
                    env.num_envs, self._last_action_masks
                )
                with th.no_grad():
                    obs_tensor = self.policy.obs_to_tensor(self._last_obs)[0]
                    _, next_lstm_states = self.policy.predict(
                        obs_tensor,
                        state=self._last_lstm_states,
                        episode_start=self._last_episode_starts,
                        deterministic=True,
                        action_masks=self._last_action_masks,
                    )
            else:
                with th.no_grad():
                    obs_tensor = self.policy.obs_to_tensor(self._last_obs)[0]

                    if self.exploration_strategy == "boltzmann":
                        actions, next_lstm_states = self.policy.predict(
                            obs_tensor,
                            state=self._last_lstm_states,
                            episode_start=self._last_episode_starts,
                            deterministic=False,
                            temperature=self.exploration_temp,
                            action_masks=self._last_action_masks,
                        )
                    else:
                        # Epsilon-greedy
                        actions, next_lstm_states = self.policy.predict(
                            obs_tensor,
                            state=self._last_lstm_states,
                            episode_start=self._last_episode_starts,
                            deterministic=True,
                            action_masks=self._last_action_masks,
                        )
                        if np.random.rand() < self.exploration_rate:
                            actions = self._sample_random_actions(
                                env.num_envs, self._last_action_masks
                            )

            # Perform action in environment
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            self._update_info_buffer(infos, dones)

            # Store transition in sequence replay buffer
            self._store_transition(
                replay_buffer, actions, new_obs, rewards, dones, infos
            )

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )
            self._on_step()

            # Save state for the next step
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = next_lstm_states
            if any("action_mask" in info for info in infos):
                next_masks = []
                for env_idx, info in enumerate(infos):
                    if dones[env_idx]:
                        try:
                            m = env.env_method(
                                "action_masks", indices=[env_idx]
                            )[0]
                        except Exception:
                            m = info.get("action_mask")
                    else:
                        m = info.get("action_mask")
                    if m is None:
                        m = np.ones(self.action_space.n, dtype=np.bool_)
                    next_masks.append(np.asarray(m, dtype=np.bool_))
                self._last_action_masks = np.asarray(
                    next_masks, dtype=np.bool_
                ).reshape(env.num_envs, self.action_space.n)

            for done in dones:
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1

        callback.on_rollout_end()
        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training=True,
        )

    def _sample_random_actions(
        self,
        n_envs: int,
        action_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if action_masks is None:
            return np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        sampled = []
        masks_arr = np.asarray(action_masks, dtype=np.bool_).reshape(
            n_envs, self.action_space.n
        )
        for i in range(n_envs):
            valid_indices = np.flatnonzero(masks_arr[i])
            if valid_indices.size > 0:
                sampled.append(int(np.random.choice(valid_indices)))
            else:
                sampled.append(int(self.action_space.sample()))
        return np.array(sampled)

    def train(self, gradient_steps: int, batch_size: int = 32) -> None:
        # Switch to training mode
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        active_seq_len = self.sequence_length + self.burn_in_steps

        losses = []
        for _ in range(gradient_steps):
            # Sample batch of sequence segments from RecurrentReplayBuffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            if isinstance(replay_data["observations"], dict):
                first_obs = next(iter(replay_data["observations"].values()))
                curr_batch_size = int(first_obs.shape[0])
                seq_len = int(first_obs.shape[1])
                flat_obs: th.Tensor | dict[str, th.Tensor] = {
                    k: v.view(-1, *self.observation_space.spaces[k].shape)
                    for k, v in replay_data["observations"].items()
                }
                flat_next_obs: th.Tensor | dict[str, th.Tensor] = {
                    k: v.view(-1, *self.observation_space.spaces[k].shape)
                    for k, v in replay_data["next_observations"].items()
                }
            else:
                curr_batch_size = int(replay_data["observations"].shape[0])
                seq_len = int(replay_data["observations"].shape[1])
                obs_shape = self.observation_space.shape
                flat_obs = replay_data["observations"].view(-1, *obs_shape)
                flat_next_obs = replay_data["next_observations"].view(
                    -1, *obs_shape
                )

            flat_episode_starts = replay_data["episode_starts"].view(-1, 1)
            flat_next_episode_starts = replay_data["dones"].view(-1, 1)

            # Initialize initial LSTM states to 0 for training step
            state_shape = (
                self.policy.n_lstm_layers,
                curr_batch_size,
                self.policy.lstm_hidden_size,
            )
            lstm_states = (
                th.zeros(*state_shape, device=self.device),
                th.zeros(*state_shape, device=self.device),
            )

            # Forward pass: current Q-values
            q_values, _ = self.q_net(
                flat_obs, lstm_states, flat_episode_starts
            )
            q_values = q_values.view(
                curr_batch_size, seq_len, self.action_space.n
            )

            # Gather Q-values for actions taken
            actions = replay_data["actions"].long()
            current_q_all = th.gather(q_values, dim=2, index=actions)

            # Compute TD Target (Double DQN / Target network evaluation)
            with th.no_grad():
                target_lstm_states = (
                    th.zeros(*state_shape, device=self.device),
                    th.zeros(*state_shape, device=self.device),
                )

                # Q-values from target network
                next_q_values, _ = self.q_net_target(
                    flat_next_obs,
                    target_lstm_states,
                    flat_next_episode_starts,
                )
                next_q_values = next_q_values.view(
                    curr_batch_size, seq_len, self.action_space.n
                )

                # Greedy actions from online network (masked for valid actions)
                online_next_q_values, _ = self.q_net(
                    flat_next_obs,
                    target_lstm_states,
                    flat_next_episode_starts,
                )
                online_next_q_values = online_next_q_values.view(
                    curr_batch_size, seq_len, self.action_space.n
                )
                if "next_action_masks" in replay_data:
                    next_masks = replay_data["next_action_masks"].bool()
                    valid_rows = next_masks.any(dim=-1, keepdim=True)
                    safe_next_masks = th.where(
                        valid_rows, next_masks, th.ones_like(next_masks)
                    )
                    online_next_q_values = online_next_q_values.masked_fill(
                        ~safe_next_masks, -1e8
                    )
                next_actions = online_next_q_values.argmax(
                    dim=-1, keepdim=True
                )

                next_q = th.gather(next_q_values, dim=2, index=next_actions)

                eval_len = min(active_seq_len, seq_len)
                if self.n_steps <= 1:
                    target_q_all = (
                        replay_data["rewards"][:, :eval_len]
                        + (1.0 - replay_data["dones"][:, :eval_len])
                        * self.gamma
                        * next_q[:, :eval_len]
                    )
                else:
                    # Exact n-step TD target truncated at episode boundaries
                    rewards_seq = replay_data["rewards"]
                    dones_seq = replay_data["dones"]
                    target_q_all = th.zeros(
                        (curr_batch_size, eval_len, 1),
                        dtype=next_q.dtype,
                        device=self.device,
                    )
                    alive = th.ones(
                        (curr_batch_size, eval_len, 1),
                        dtype=next_q.dtype,
                        device=self.device,
                    )
                    last_valid_next_q = next_q[:, :eval_len]
                    max_k = min(self.n_steps, seq_len - eval_len + 1)
                    for k in range(max_k):
                        r_k = rewards_seq[:, k:k + eval_len]
                        d_k = dones_seq[:, k:k + eval_len]
                        target_q_all = (
                            target_q_all + (self.gamma ** k) * alive * r_k
                        )
                        alive = alive * (1.0 - d_k)
                        last_valid_next_q = next_q[:, k:k + eval_len]
                    target_q_all = (
                        target_q_all
                        + (self.gamma ** max_k) * alive * last_valid_next_q
                    )

            current_q = current_q_all[:, :eval_len]
            target_q = target_q_all

            # Apply burn-in period to exclude initial steps from backprop loss
            if self.burn_in_steps > 0:
                current_q = current_q[:, self.burn_in_steps:]
                target_q = target_q[:, self.burn_in_steps:]

            # Huber loss (compute in FP32 to prevent underflow/overflow)
            loss = th.nn.functional.smooth_l1_loss(
                current_q.float(), target_q.float()
            )
            if not th.isfinite(loss):
                warnings.warn(
                    "Non-finite loss encountered in DDRQN.train(); "
                    "skipping gradient step."
                )
                self.policy.optimizer.zero_grad(set_to_none=True)
                continue

            # Optimize weights outside any potential AMP autocast context
            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.max_grad_norm is not None:
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                if not th.isfinite(grad_norm):
                    warnings.warn(
                        "Non-finite gradient norm encountered in "
                        "DDRQN.train(); skipping optimizer step."
                    )
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    continue
            else:
                has_non_finite_grad = any(
                    p.grad is not None and not th.isfinite(p.grad).all()
                    for p in self.policy.parameters()
                )
                if has_non_finite_grad:
                    warnings.warn(
                        "Non-finite parameter gradient encountered in "
                        "DDRQN.train(); skipping optimizer step."
                    )
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    continue

            self.policy.optimizer.step()
            losses.append(loss.item())

        self._n_updates += gradient_steps

        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record(
            "train/loss", float(np.mean(losses)) if losses else 0.0
        )

    def predict(
        self,
        observation: Any,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides predict to correctly handle recurrence parameters and
        optional action_masks.
        """
        if not deterministic:
            if self.exploration_strategy == "boltzmann":
                action, state = self.policy.predict(
                    observation,
                    state,
                    episode_start,
                    deterministic=False,
                    temperature=self.exploration_temp,
                    action_masks=action_masks,
                )
            else:
                # Epsilon-greedy
                if np.random.rand() < self.exploration_rate:
                    if self.policy.is_vectorized_observation(observation):
                        if isinstance(observation, dict):
                            n_batch = int(
                                next(iter(observation.values())).shape[0]
                            )
                        else:
                            n_batch = int(observation.shape[0])
                        action = self._sample_random_actions(
                            n_batch, action_masks
                        )
                    else:
                        action = self._sample_random_actions(
                            1, action_masks
                        ).squeeze(axis=0)
                    # Run policy forward pass to maintain recurrent states
                    _, state = self.policy.predict(
                        observation,
                        state,
                        episode_start,
                        deterministic=True,
                        action_masks=action_masks,
                    )
                else:
                    action, state = self.policy.predict(
                        observation,
                        state,
                        episode_start,
                        deterministic=True,
                        action_masks=action_masks,
                    )
        else:
            action, state = self.policy.predict(
                observation,
                state,
                episode_start,
                deterministic=True,
                action_masks=action_masks,
            )

        return action, state

    def learn(
        self: SelfDDRQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDRQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDRQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "q_net",
            "q_net_target",
            "_last_lstm_states",
            "_last_action_masks",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
