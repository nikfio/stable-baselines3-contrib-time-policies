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


class DummyDict4ActionEnv(gym.Env):
    """
    Dummy environment with Pattern A Dict({"market": (4, 5, 3), "portfolio": (5,)})
    observations and 4 discrete actions [SELL, BUY, HOLD, CLOSE] with
    state-dependent action masking.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "market": spaces.Box(
                    low=-1.0, high=1.0, shape=(4, 5, 3), dtype=np.float32
                ),
                "portfolio": spaces.Box(
                    low=-5.0, high=5.0, shape=(5,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(4)
        self._step = 0
        self._in_pos = False
        self.max_steps = 12

    def action_masks(self) -> np.ndarray:
        if self._in_pos:
            return np.array([False, False, True, True], dtype=np.bool_)
        return np.array([True, True, True, False], dtype=np.bool_)

    def _get_obs(self):
        pos_dir = 1.0 if self._in_pos else 0.0
        is_anchor = 0.0 if self._in_pos else 1.0
        return {
            "market": self.observation_space.spaces["market"].sample(),
            "portfolio": np.array(
                [pos_dir, 0.25, -0.1, 0.3, is_anchor], dtype=np.float32
            ),
        }

    def reset(self, *, seed=None, options=None):
        self._step = 0
        self._in_pos = False
        return self._get_obs(), {"action_mask": self.action_masks()}

    def step(self, action):
        self._step += 1
        act = int(action)
        if not self._in_pos and act in (0, 1):
            self._in_pos = True
        elif self._in_pos and act == 3:
            self._in_pos = False
        obs = self._get_obs()
        reward = 0.5 if act in (1, 3) else -0.1
        terminated = self._step >= self.max_steps
        truncated = False
        return (
            obs,
            reward,
            terminated,
            truncated,
            {"action_mask": self.action_masks()},
        )


def test_ddrqn_lily_5_6_dict_obs_masking_nstep_and_layernorm_silu():
    """
    Test Lily-5.6 features in DDRQN:
      - Dict({"market", "portfolio"}) observation space with skip-connection
      - float16 storage for 3D market observations in RecurrentReplayBuffer
      - 4-action masking in predict(), collect_rollouts(), and Double-DQN train()
      - Multi-step TD targets (n_steps=3)
      - LayerNorm + SiLU activation in TimeCNN and DdrqnQNetwork
      - forward_actor_and_value() hook for model decomposition
    """
    env = DummyDict4ActionEnv()
    model = DDRQN(
        policy=TimeCnnLstmDdrqnPolicy,
        env=env,
        learning_rate=1e-3,
        buffer_size=300,
        replay_buffer_storage_dtype="float16",
        learning_starts=16,
        batch_size=4,
        train_freq=4,
        gradient_steps=2,
        exploration_strategy="boltzmann",
        exploration_initial_temp=1.0,
        exploration_final_temp=0.1,
        sequence_length=4,
        burn_in_steps=2,
        n_steps=3,
        policy_kwargs={
            "use_layer_norm": True,
            "activation_fn": "silu",
            "features_extractor_kwargs": {
                "features_dim": 32,
                "cnn_layers": [32, 32],
                "enable_fc": True,
                "fc_layers": [16],
            },
            "lstm_hidden_size": 32,
            "n_lstm_layers": 2,
            "val_head_arch": [32, 16],
            "adv_head_arch": [32, 16],
        },
        seed=99,
    )

    # Check FP16 storage for market and FP32 for portfolio
    assert model.replay_buffer.observations["market"].dtype == np.float16
    assert model.replay_buffer.observations["portfolio"].dtype == np.float32
    assert model.policy.q_net.portfolio_dim == 5
    assert model.policy.q_net.lstm.input_size == 32 + 5

    # Check action masking on FLAT mask [True, True, True, False]
    obs, _ = env.reset()
    flat_mask = np.array([[True, True, True, False]], dtype=np.bool_)
    for _ in range(20):
        act_det, _ = model.predict(
            obs, deterministic=True, action_masks=flat_mask
        )
        assert int(act_det) in (0, 1, 2)
        act_stoch, _ = model.predict(
            obs, deterministic=False, action_masks=flat_mask
        )
        assert int(act_stoch) in (0, 1, 2)

    # Check action masking on IN-POSITION mask [False, False, True, True]
    pos_mask = np.array([[False, False, True, True]], dtype=np.bool_)
    for _ in range(20):
        act_det, _ = model.predict(
            obs, deterministic=True, action_masks=pos_mask
        )
        assert int(act_det) in (2, 3)
        act_stoch, _ = model.predict(
            obs, deterministic=False, action_masks=pos_mask
        )
        assert int(act_stoch) in (2, 3)

    # Run learning with n_steps=3 and verify sampled batch casts market to float32
    model.learn(total_timesteps=48)
    sample = model.replay_buffer.sample(batch_size=2)
    assert sample["observations"]["market"].dtype == th.float32
    assert sample["next_action_masks"].dtype == th.bool

    # Verify stateless forward_actor_and_value hook works with raw 4D market tensor
    raw_market_tensor = th.randn(4, 4, 5, 3, device=model.device)
    q_vals, val = model.policy.q_net.forward_actor_and_value(raw_market_tensor)
    assert q_vals.shape == (4, 4)
    assert val.shape == (4, 1)
