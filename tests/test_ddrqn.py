import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
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
    
    # Verify the layers
    assert hasattr(model.policy, "q_net")
    assert hasattr(model.policy, "q_net_target")
    
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
    assert len(state) == 2 # LSTM (h, c)
    assert state[0].shape == (1, 1, model.policy.lstm_hidden_size) # (n_layers, n_envs, lstm_hidden_size)
    
    # Clean up file
    import os
    if os.path.exists("test_ddrqn_model.zip"):
        os.remove("test_ddrqn_model.zip")
