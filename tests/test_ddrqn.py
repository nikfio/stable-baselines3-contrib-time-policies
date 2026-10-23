import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch as th
from sb3_contrib import DDRQN
from sb3_contrib.ddrqn.policies import TimeCnnLstmDdrqnPolicy


class Dummy3DEnv(gym.Env):
    """
    Dummy environment with a 3D observation space (horizon, candles, features)
    and discrete actions.
    """
    def __init__(self):
        super().__init__()
        # 3D observation shape: e.g., (4, 5, 3)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4, 5, 3), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._step = 0
        self.max_steps = 10

    def reset(self, *, seed=None, options=None):
        self._step = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._step += 1
        obs = self.observation_space.sample()
        reward = 1.0 if action == 1 else 0.0
        terminated = self._step >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_ddrqn_training_and_save_load():
    env = Dummy3DEnv()

    # 1. Initialize DDRQN model
    model = DDRQN(
        policy="TimeCnnLstmDdrqnPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=1000,
        learning_starts=20,
        batch_size=4,
        tau=0.05,
        gamma=0.99,
        train_freq=2,
        gradient_steps=1,
        target_update_interval=50,
        sequence_length=4,
        burn_in_steps=2,
        seed=42,
    )

    # Verify the layers and initial exploration schedules
    assert hasattr(model.policy, "q_net")
    assert hasattr(model.policy, "q_net_target")
    assert model.exploration_temp > 0.0
    assert model.exploration_rate > 0.0

    # 2. Train the model for a few steps
    model.learn(total_timesteps=40)

    # 3. Test saving the model
    model.save("test_ddrqn_model")

    # 4. Test loading the model
    loaded_model = DDRQN.load("test_ddrqn_model", env=env)
    assert loaded_model.sequence_length == 4
    assert loaded_model.burn_in_steps == 2

    # 5. Run prediction
    obs, _ = env.reset()
    action, state = loaded_model.predict(obs, deterministic=True)
    assert action is not None
    assert state is not None
    assert len(state) == 2  # LSTM (h, c)
    assert state[0].shape == (1, 1, model.policy.lstm_hidden_size)

    # Clean up file
    if os.path.exists("test_ddrqn_model.zip"):
        os.remove("test_ddrqn_model.zip")


def test_ddrqn_boltzmann_nan_and_zero_temp_resilience():
    """
    Verify that TimeCnnLstmDdrqnPolicy.predict and DDRQN.train gracefully
    handle zero/negative temperatures, NaN/Inf Q-values, and non-finite
    gradients without raising RuntimeError or corrupting weights.
    """
    env = Dummy3DEnv()
    model = DDRQN(
        policy=TimeCnnLstmDdrqnPolicy,
        env=env,
        learning_rate=1e-3,
        buffer_size=200,
        learning_starts=10,
        batch_size=4,
        train_freq=4,
        gradient_steps=2,
        exploration_strategy="boltzmann",
        exploration_initial_temp=1.0,
        exploration_final_temp=0.05,
        sequence_length=4,
        burn_in_steps=2,
        seed=123,
    )

    obs, _ = env.reset()

    # 1. Predict with temperature=0.0 (previously caused div by zero -> NaN)
    action, state = model.policy.predict(
        obs, deterministic=False, temperature=0.0
    )
    assert action in [0, 1, 2]
    assert np.isfinite(state[0]).all()
    assert np.isfinite(state[1]).all()

    # 2. Inject NaN into value_head bias and verify stochastic predict works
    orig_bias = model.policy.q_net.value_head[-1].bias.data.clone()
    with th.no_grad():
        model.policy.q_net.value_head[-1].bias.fill_(float("nan"))

    action_nan, state_nan = model.policy.predict(
        obs, deterministic=False, temperature=0.1
    )
    assert action_nan in [0, 1, 2]
    assert np.isfinite(state_nan[0]).all()

    # Restore valid weights
    with th.no_grad():
        model.policy.q_net.value_head[-1].bias.copy_(orig_bias)

    # 3. Collect steps and verify non-finite loss/grad guard skips update
    model.learn(total_timesteps=20)
    weights_before = {
        k: v.clone() for k, v in model.policy.q_net.state_dict().items()
    }

    # Corrupt replay buffer rewards with NaN and run train()
    model.replay_buffer.rewards[:, :] = np.nan
    with pytest.warns(UserWarning, match="Non-finite"):
        model.train(gradient_steps=1, batch_size=4)

    # Verify network weights remain completely finite and uncorrupted
    for k, v in model.policy.q_net.state_dict().items():
        assert th.isfinite(v).all(), f"Parameter {k} became non-finite!"
        assert th.allclose(v, weights_before[k])
