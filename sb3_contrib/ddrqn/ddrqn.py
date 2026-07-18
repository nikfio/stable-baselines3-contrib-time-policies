import warnings
from typing import Any, ClassVar, Optional, Type, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, TrainFreq
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.ddrqn.buffers import RecurrentReplayBuffer
from sb3_contrib.ddrqn.policies import TimeCnnLstmDdrqnPolicy, DdrqnQNetwork

SelfDDRQN = TypeVar("SelfDDRQN", bound="DDRQN")

class DDRQN(OffPolicyAlgorithm):
    """
    Dueling Deep Recurrent Q-Network (DDRQN)
    Specifically tailored for recurrent sequence learning over time-series inputs.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, TimeCnnLstmDdrqnPolicy, ...)
    :param env: The environment to learn from
    :param learning_rate: The learning rate (float or Schedule)
    :param buffer_size: Size of the replay buffer (number of transitions)
    :param learning_starts: How many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size (number of sequences) for each gradient update
    :param tau: The soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: The discount factor
    :param train_freq: Update the model every ``train_freq`` steps
    :param gradient_steps: How many gradient steps to do after each rollout
    :param target_update_interval: Update target network every ``target_update_interval`` env steps
    :param exploration_fraction: Fraction of entire training period over which exploration rate decays
    :param exploration_initial_eps: Initial value of random action probability
    :param exploration_final_eps: Final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param sequence_length: Sequence length for Backpropagation Through Time (BPTT)
    :param burn_in_steps: Number of initial steps in sampled sequences to warm up recurrent hidden states
    :param stats_window_size: Window size for the rollout logging
    :param tensorboard_log: The log location for tensorboard
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
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
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # We need a custom replay buffer that returns sequences
        replay_buffer_kwargs = {"sequence_length": sequence_length + burn_in_steps}

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
        self.exploration_rate = 0.0
        self.exploration_temp = 0.0
        self._last_lstm_states = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats for batch norm (if any)
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        
        from stable_baselines3.common.utils import LinearSchedule
        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )
        self.temp_schedule = LinearSchedule(
            self.exploration_initial_temp, self.exploration_final_temp, self.exploration_fraction
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
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Polyak updates of batch norm stats
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.exploration_temp = self.temp_schedule(self._current_progress_remaining)
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
        Custom collect_rollouts to track and update the recurrent LSTM hidden states.
        """
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        callback.on_rollout_start()

        # Initialize or maintain LSTM states
        if self._last_lstm_states is None:
            self._last_lstm_states = (
                np.zeros((self.policy.n_lstm_layers, env.num_envs, self.policy.lstm_hidden_size), dtype=np.float32),
                np.zeros((self.policy.n_lstm_layers, env.num_envs, self.policy.lstm_hidden_size), dtype=np.float32)
            )

        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.num_timesteps < learning_starts:
                actions = np.array([self.action_space.sample() for _ in range(env.num_envs)])
                with th.no_grad():
                    obs_tensor = self.policy.obs_to_tensor(self._last_obs)[0]
                    _, next_lstm_states = self.policy.predict(
                        obs_tensor,
                        state=self._last_lstm_states,
                        episode_start=self._last_episode_starts,
                        deterministic=True
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
                            temperature=self.exploration_temp
                        )
                    else:
                        # Epsilon-greedy
                        actions, next_lstm_states = self.policy.predict(
                            obs_tensor,
                            state=self._last_lstm_states,
                            episode_start=self._last_episode_starts,
                            deterministic=True
                        )
                        if np.random.rand() < self.exploration_rate:
                            actions = np.array([self.action_space.sample() for _ in range(env.num_envs)])

            # Perform action in environment
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            self._update_info_buffer(infos, dones)

            # Store transition in sequence replay buffer
            self._store_transition(replay_buffer, actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            # Save state for the next step
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = next_lstm_states

            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1

        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=True)

    def train(self, gradient_steps: int, batch_size: int = 32) -> None:
        # Switch to training mode
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample batch of sequence segments from RecurrentReplayBuffer
            # Shape of observations inside replay_data: (batch_size * n_envs, sequence_length + burn_in_steps, *obs_shape)
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            curr_batch_size = replay_data["observations"].shape[0]
            seq_len = replay_data["observations"].shape[1]

            # Flatten temporal and batch dimensions for TimeCNN features extractor
            flat_obs = replay_data["observations"].view(-1, *self.observation_space.shape)
            flat_next_obs = replay_data["next_observations"].view(-1, *self.observation_space.shape)
            flat_episode_starts = replay_data["episode_starts"].view(-1, 1)

            # Initialize initial LSTM states to 0 for training step
            lstm_states = (
                th.zeros(self.policy.n_lstm_layers, curr_batch_size, self.policy.lstm_hidden_size, device=self.device),
                th.zeros(self.policy.n_lstm_layers, curr_batch_size, self.policy.lstm_hidden_size, device=self.device)
            )

            # Forward pass: current Q-values
            q_values, _ = self.q_net(flat_obs, lstm_states, flat_episode_starts)
            q_values = q_values.view(curr_batch_size, seq_len, self.action_space.n)

            # Gather Q-values for actions taken
            actions = replay_data["actions"].long()
            current_q = th.gather(q_values, dim=2, index=actions)

            # Compute TD Target (Double DQN / Target network evaluation)
            with th.no_grad():
                target_lstm_states = (
                    th.zeros(self.policy.n_lstm_layers, curr_batch_size, self.policy.lstm_hidden_size, device=self.device),
                    th.zeros(self.policy.n_lstm_layers, curr_batch_size, self.policy.lstm_hidden_size, device=self.device)
                )
                
                # Q-values from target network
                next_q_values, _ = self.q_net_target(flat_next_obs, target_lstm_states, flat_episode_starts)
                next_q_values = next_q_values.view(curr_batch_size, seq_len, self.action_space.n)

                # Greedy actions from online network
                online_next_q_values, _ = self.q_net(flat_next_obs, target_lstm_states, flat_episode_starts)
                online_next_q_values = online_next_q_values.view(curr_batch_size, seq_len, self.action_space.n)
                next_actions = online_next_q_values.argmax(dim=-1, keepdim=True)

                next_q = th.gather(next_q_values, dim=2, index=next_actions)
                
                # y_t = r_t + gamma * (1 - done_t) * Q_target(s_t+1, argmax Q_online)
                target_q = replay_data["rewards"] + (1.0 - replay_data["dones"]) * self.gamma * next_q

            # Apply burn-in period to exclude initial steps from backprop loss
            if self.burn_in_steps > 0:
                current_q = current_q[:, self.burn_in_steps:]
                target_q = target_q[:, self.burn_in_steps:]

            # Huber loss
            loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())

            # Optimize weights
            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides predict to correctly handle recurrence parameters.
        """
        if not deterministic:
            if self.exploration_strategy == "boltzmann":
                action, state = self.policy.predict(
                    observation,
                    state,
                    episode_start,
                    deterministic=False,
                    temperature=self.exploration_temp
                )
            else:
                # Epsilon-greedy
                if np.random.rand() < self.exploration_rate:
                    if self.policy.is_vectorized_observation(observation):
                        n_batch = observation.shape[0]
                        action = np.array([self.action_space.sample() for _ in range(n_batch)])
                    else:
                        action = np.array(self.action_space.sample())
                    # Run policy forward pass anyway to maintain the recurrent states
                    _, state = self.policy.predict(observation, state, episode_start, deterministic=True)
                else:
                    action, state = self.policy.predict(observation, state, episode_start, deterministic=True)
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic=True)

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
        return super()._excluded_save_params() + ["q_net", "q_net_target", "_last_lstm_states"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
