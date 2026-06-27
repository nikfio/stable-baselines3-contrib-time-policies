import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies_LoRA import TimeCnnLstmPolicy


class DummyTimeSeriesEnv(gym.Env):
    """
    Dummy environment simulating a time-series/forex scenario
    with observation shape (sequence_length, channels, features).
    """

    def __init__(self):
        super().__init__()
        # 12 steps, 5 channels, 4 features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12, 5, 4), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._step += 1
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = self._step >= 20
        truncated = False
        return obs, reward, terminated, truncated, {}


@pytest.mark.parametrize("use_lora", [True, False])
def test_time_cnn_lstm_policy(use_lora):
    env = DummyTimeSeriesEnv()

    policy_kwargs = dict(
        use_lora=use_lora,
        lora_rank=4,
        lora_alpha=8,
        enable_critic_lstm=True,
    )

    model = RecurrentPPO(
        TimeCnnLstmPolicy,
        env,
        n_steps=16,
        seed=0,
        policy_kwargs=policy_kwargs,
        n_epochs=2,
        batch_size=8,
    )

    # 1. Verify model and parameter properties before learning
    policy = model.policy

    if use_lora:
        # Check that original weights are parametrized and frozen
        assert hasattr(policy.lstm_actor, "parametrizations")
        assert "weight_ih_l0" in policy.lstm_actor.parametrizations
        assert "weight_hh_l0" in policy.lstm_actor.parametrizations

        # Original parameters must have requires_grad=False
        actor_params = policy.lstm_actor.parametrizations
        orig_weight_ih = actor_params.weight_ih_l0.original
        orig_weight_hh = actor_params.weight_hh_l0.original
        assert not orig_weight_ih.requires_grad
        assert not orig_weight_hh.requires_grad

        # LoRA adapter parameters must have requires_grad=True
        lora_A_ih = policy.lstm_actor.parametrizations.weight_ih_l0[0].lora_A
        lora_B_ih = policy.lstm_actor.parametrizations.weight_ih_l0[0].lora_B
        assert lora_A_ih.requires_grad
        assert lora_B_ih.requires_grad

        # Same for critic LSTM
        critic_params = policy.lstm_critic.parametrizations
        assert hasattr(policy.lstm_critic, "parametrizations")
        assert not critic_params.weight_ih_l0.original.requires_grad
        assert critic_params.weight_ih_l0[0].lora_A.requires_grad

        # Check CNN Conv2d parametrization
        cnn_conv = policy.features_extractor.cnn[0]
        assert hasattr(cnn_conv, "parametrizations")
        assert not cnn_conv.parametrizations.weight.original.requires_grad
        assert cnn_conv.parametrizations.weight[0].lora_A.requires_grad

        # Check MLP head parametrization (action_net)
        assert hasattr(policy.action_net, "parametrizations")
        act_net_params = policy.action_net.parametrizations.weight
        assert not act_net_params.original.requires_grad
        assert act_net_params[0].lora_A.requires_grad
    else:
        # Check that traditional training mode does not use parametrization
        assert not hasattr(policy.lstm_actor, "parametrizations")
        assert policy.lstm_actor.weight_ih_l0.requires_grad
        assert policy.lstm_actor.weight_hh_l0.requires_grad

        if policy.lstm_critic is not None:
            assert not hasattr(policy.lstm_critic, "parametrizations")
            assert policy.lstm_critic.weight_ih_l0.requires_grad

    # 2. Run learning loop to verify optimization step runs without crashes
    model.learn(total_timesteps=32)

    # 3. Check parameter updates (non-zero gradients or change value)
    if use_lora:
        # Check that gradients were calculated for LoRA parameters
        lora_A_ih = policy.lstm_actor.parametrizations.weight_ih_l0[0].lora_A
        assert lora_A_ih.grad is not None
        assert torch.norm(lora_A_ih.grad) > 0.0

        # Original parameters should NOT have gradients computed
        orig_weight_ih = (
            policy.lstm_actor.parametrizations.weight_ih_l0.original
        )
        assert orig_weight_ih.grad is None

        # Check that gradients were calculated for CNN LoRA parameters
        cnn_conv = policy.features_extractor.cnn[0]
        lora_A_cnn = cnn_conv.parametrizations.weight[0].lora_A
        assert lora_A_cnn.grad is not None
        assert torch.norm(lora_A_cnn.grad) > 0.0
    else:
        # Standard weights should have gradients
        assert policy.lstm_actor.weight_ih_l0.grad is not None
        assert torch.norm(policy.lstm_actor.weight_ih_l0.grad) > 0.0
