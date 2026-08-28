"""Recurrent actor-critic used by the direct-action PPO baseline."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PPOActorCritic(nn.Module):
    """Depth/goal policy with the same encoder capacity as ``model_mpc.Model``.

    The stochastic policy is defined in an unconstrained latent space.  A tanh
    transform maps samples to [-1, 1]; linear speed is then mapped to
    [0, max_v] and angular speed to [-max_omega, max_omega].
    """

    def __init__(
        self,
        dim_obs=6,
        hidden_dim=192,
        input_w=32,
        input_h=24,
        max_v=2.0,
        max_omega=3.0,
        initial_speed=1.5,
        initial_log_std=-0.7,
    ):
        super().__init__()
        if max_v <= 0.0 or max_omega <= 0.0:
            raise ValueError("Velocity limits must be positive")
        if not 0.0 < initial_speed < max_v:
            raise ValueError("initial_speed must be in (0, max_v)")

        self.max_v = float(max_v)
        self.max_omega = float(max_omega)

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, kernel_size=3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, kernel_size=3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_h, input_w)
            conv_out_dim = self.conv_net(dummy).shape[1]

        self.fc_visual = nn.Linear(conv_out_dim, 192, bias=False)
        self.fc_state = nn.Linear(dim_obs, 64)
        self.fc_state.weight.data.mul_(0.5)
        self.gru = nn.GRUCell(192 + 64, hidden_dim)

        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 1),
        )
        self.actor_mean[2].weight.data.mul_(0.01)
        speed_normalized = 2.0 * initial_speed / max_v - 1.0
        speed_normalized = min(max(speed_normalized, -0.999), 0.999)
        speed_latent = math.atanh(speed_normalized)
        self.actor_mean[2].bias.data.copy_(
            torch.tensor([speed_latent, 0.0], dtype=torch.float32)
        )
        self.critic[2].weight.data.mul_(0.01)
        self.critic[2].bias.data.zero_()
        self.log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))

    def reset(self):
        """Compatibility with the existing evaluation pipeline."""

    def forward(self, depth, state, hidden=None):
        batch_size = depth.shape[0]
        visual = self.fc_visual(self.conv_net(depth))
        state_feature = F.leaky_relu(self.fc_state(state), 0.05)
        fused = torch.cat([visual, state_feature], dim=1)
        if hidden is None:
            hidden = torch.zeros(
                batch_size,
                self.gru.hidden_size,
                device=depth.device,
                dtype=depth.dtype,
            )
        hidden = self.gru(fused, hidden)
        mean = self.actor_mean(hidden)
        value = self.critic(hidden).squeeze(-1)
        return mean, value, hidden

    def _distribution(self, mean):
        log_std = self.log_std.clamp(-5.0, 1.0)
        return Normal(mean, log_std.exp().expand_as(mean))

    @staticmethod
    def _squashed_log_prob(distribution, latent, normalized_action):
        correction = torch.log(1.0 - normalized_action.square() + 1e-6)
        return (distribution.log_prob(latent) - correction).sum(dim=-1)

    def normalized_to_command(self, normalized_action):
        linear = 0.5 * (normalized_action[:, 0:1] + 1.0) * self.max_v
        angular = normalized_action[:, 1:2] * self.max_omega
        return torch.cat([linear, angular], dim=1)

    def sample_action(self, depth, state, hidden=None):
        mean, value, hidden = self(depth, state, hidden)
        distribution = self._distribution(mean)
        latent = distribution.rsample()
        normalized_action = torch.tanh(latent)
        log_prob = self._squashed_log_prob(
            distribution, latent, normalized_action
        )
        command = self.normalized_to_command(normalized_action)
        return command, latent, log_prob, value, hidden

    def deterministic_action(self, depth, state, hidden=None):
        mean, _, hidden = self(depth, state, hidden)
        command = self.normalized_to_command(torch.tanh(mean))
        return command, hidden

    def evaluate_latent(self, depth, state, latent, hidden=None):
        mean, value, hidden = self(depth, state, hidden)
        distribution = self._distribution(mean)
        normalized_action = torch.tanh(latent)
        log_prob = self._squashed_log_prob(
            distribution, latent, normalized_action
        )
        # The base-normal entropy is the usual stable approximation for a
        # tanh-squashed Gaussian policy.
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value, hidden
