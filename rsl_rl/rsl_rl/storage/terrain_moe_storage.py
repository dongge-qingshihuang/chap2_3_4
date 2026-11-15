import torch


class TerrainMoeStorage:
    """Rollout storage for the Go2 predictive-expert method."""

    class Transition:
        pass

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        dynamics_target_dim,
        contact_dim,
        terrain_dim,
        future_contact_steps,
        device,
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0

        def zeros(shape):
            return torch.zeros(
                num_transitions_per_env, num_envs, *shape, device=device
            )

        self.observations = zeros(actor_obs_shape)
        self.critic_observations = zeros(critic_obs_shape)
        self.actions = zeros(action_shape)
        self.rewards = zeros([1])
        self.dones = zeros([1]).bool()
        self.values = zeros([1])
        self.returns = zeros([1])
        self.advantages = zeros([1])
        self.actions_log_prob = zeros([1])
        self.mu = zeros(action_shape)
        self.sigma = zeros(action_shape)
        self.next_dynamics = zeros([dynamics_target_dim])
        self.contact_targets = zeros([contact_dim])
        self.future_contact_targets = torch.zeros(
            num_transitions_per_env,
            num_envs,
            future_contact_steps,
            contact_dim,
            device=device,
        )
        self.future_contact_valid = torch.zeros(
            num_transitions_per_env,
            num_envs,
            future_contact_steps,
            1,
            dtype=torch.bool,
            device=device,
        )
        self.next_contact = self.future_contact_targets
        self.terrain_heights = zeros([terrain_dim])
        self.transition = self.Transition()

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("TerrainMoeStorage overflow")
        self.observations[self.step].copy_(transition.observations)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values.view(-1, 1))
        self.actions_log_prob[self.step].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[self.step].copy_(transition.mu)
        self.sigma[self.step].copy_(transition.sigma)
        self.next_dynamics[self.step].copy_(transition.next_dynamics)
        self.contact_targets[self.step].copy_(transition.next_contact)
        self.terrain_heights[self.step].copy_(transition.terrain_heights)
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + gamma * next_values * next_is_not_terminal
                - self.values[step]
            )
            advantage = (
                delta + gamma * lam * next_is_not_terminal * advantage
            )
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (
            self.advantages - self.advantages.mean()
        ) / (self.advantages.std() + 1e-8)

    def build_future_contacts(self, horizon):
        steps, envs, contact_dim = self.contact_targets.shape
        if horizon > self.future_contact_targets.size(2):
            raise ValueError(
                f"future contact horizon {horizon} exceeds storage capacity "
                f"{self.future_contact_targets.size(2)}"
            )
        future = torch.zeros(
            steps,
            envs,
            horizon,
            contact_dim,
            device=self.contact_targets.device,
            dtype=self.contact_targets.dtype,
        )
        valid = torch.zeros(
            steps,
            envs,
            horizon,
            1,
            dtype=torch.bool,
            device=self.contact_targets.device,
        )
        done_prefix = torch.cat(
            (
                torch.zeros(
                    1,
                    envs,
                    dtype=torch.long,
                    device=self.contact_targets.device,
                ),
                self.dones.squeeze(-1).long().cumsum(dim=0),
            ),
            dim=0,
        )
        for offset in range(horizon):
            if offset == 0:
                future[:, :, 0] = self.contact_targets
                valid[:, :, 0, 0] = True
                continue
            available = steps - offset
            if available <= 0:
                break
            target_in_same_episode = (
                done_prefix[offset + 1 :]
                - done_prefix[: -(offset + 1)]
                == 0
            )
            valid[:available, :, offset, 0] = target_in_same_episode
            future[:available, :, offset] = (
                self.contact_targets[offset:]
                * target_in_same_episode.unsqueeze(-1)
            )
        self.future_contact_targets = future
        self.future_contact_valid = valid
        self.next_contact = self.future_contact_targets

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        def flatten(value):
            return value.flatten(0, 1)

        observations = flatten(self.observations)
        critic_observations = flatten(self.critic_observations)
        actions = flatten(self.actions)
        values = flatten(self.values)
        advantages = flatten(self.advantages)
        returns = flatten(self.returns)
        old_actions_log_prob = flatten(self.actions_log_prob)
        old_mu = flatten(self.mu)
        old_sigma = flatten(self.sigma)
        next_dynamics = flatten(self.next_dynamics)
        next_contact = self.next_contact.flatten(0, 1).reshape(
            batch_size, -1
        )
        contact_valid = self.future_contact_valid.flatten(0, 1).squeeze(-1)
        terrain_heights = flatten(self.terrain_heights)
        dones = flatten(self.dones)

        for _ in range(num_epochs):
            for start in range(0, batch_size, mini_batch_size):
                batch_idx = indices[start : start + mini_batch_size]
                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    next_dynamics[batch_idx],
                    next_contact[batch_idx],
                    contact_valid[batch_idx],
                    terrain_heights[batch_idx],
                    1.0 - dones[batch_idx].float(),
                )

    def clear(self):
        self.step = 0
