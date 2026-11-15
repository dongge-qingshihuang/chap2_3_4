"""Load-aware critic advantage fusion utilities."""

import torch
import torch.nn as nn


class LoadAwareWeightNetwork(nn.Module):
    """Map a normalized 9-D payload estimate to a critic simplex."""

    def __init__(
        self,
        payload_dim=9,
        hidden_dim=64,
        min_weight=0.05,
        max_weight=0.90,
        num_critics=3,
    ):
        super().__init__()
        payload_dim = int(payload_dim)
        hidden_dim = int(hidden_dim)
        num_critics = int(num_critics)
        min_weight = float(min_weight)
        max_weight = float(max_weight)
        if payload_dim != 9:
            raise ValueError(
                f"LoadAwareWeightNetwork expects a 9-D payload, got {payload_dim}"
            )
        if hidden_dim < 1 or num_critics < 1:
            raise ValueError("hidden_dim and num_critics must be positive")
        if not 0.0 <= min_weight < 1.0:
            raise ValueError("min_weight must be in [0, 1)")
        if not min_weight < max_weight <= 1.0:
            raise ValueError("max_weight must satisfy min_weight < max_weight <= 1")
        if num_critics * min_weight >= 1.0:
            raise ValueError("num_critics * min_weight must be below 1")
        natural_max = 1.0 - (num_critics - 1) * min_weight
        if max_weight + 1e-8 < natural_max:
            raise ValueError(
                "max_weight is incompatible with the requested minimum: "
                f"max_weight={max_weight}, required >= {natural_max:.6f}"
            )

        self.payload_dim = payload_dim
        self.num_critics = num_critics
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.net = nn.Sequential(
            nn.Linear(payload_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_critics),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, payload_estimate, return_logits=False):
        if payload_estimate.shape[-1] != self.payload_dim:
            raise ValueError(
                "payload_estimate must end in 9 dimensions, got "
                f"{tuple(payload_estimate.shape)}"
            )
        logits = self.net(payload_estimate)
        probabilities = torch.softmax(logits, dim=-1)
        weights = self.min_weight + (
            1.0 - self.num_critics * self.min_weight
        ) * probabilities
        if return_logits:
            return weights, logits, probabilities
        return weights


def load_risk_components(
    payload_state,
    position_coef=1.0,
    torque_coef=1.0,
    velocity_coef=1.0,
):
    """Compute normalized COM, gravity-torque, and COM-velocity risk terms.

    ``payload_state`` is already normalized by the environment with configured
    physical scales, so the three blocks can be combined without mixing raw
    metre, newton-metre, and metre-per-second units.
    """
    if payload_state.shape[-1] != 9:
        raise ValueError("payload_state must have 9 dimensions")
    position = payload_state[..., 0:3].norm(dim=-1)
    torque = payload_state[..., 3:6].norm(dim=-1)
    velocity = payload_state[..., 6:9].norm(dim=-1)
    risk = (
        float(position_coef) * position
        + float(torque_coef) * torque
        + float(velocity_coef) * velocity
    )
    return {
        "risk": risk,
        "position": position,
        "torque": torque,
        "velocity": velocity,
    }


def physical_prior_weights(
    payload_state,
    stability_min=0.20,
    stability_max=0.70,
    risk_threshold=0.5,
    risk_slope=5.0,
    task_energy_ratio=0.8,
    position_coef=1.0,
    torque_coef=1.0,
    velocity_coef=1.0,
):
    """Build interpretable supervision targets for load-aware fusion.

    These weights are a physical prior, not an optimal-policy target. Higher
    instantaneous load risk increases stability priority; the remaining mass
    is split between task and energy using ``task_energy_ratio``.
    """
    stability_min = float(stability_min)
    stability_max = float(stability_max)
    task_energy_ratio = float(task_energy_ratio)
    if not 0.0 <= stability_min <= stability_max <= 1.0:
        raise ValueError("Require 0 <= stability_min <= stability_max <= 1")
    if not 0.0 <= task_energy_ratio <= 1.0:
        raise ValueError("task_energy_ratio must be in [0, 1]")

    components = load_risk_components(
        payload_state,
        position_coef=position_coef,
        torque_coef=torque_coef,
        velocity_coef=velocity_coef,
    )
    stability_ratio = torch.sigmoid(
        float(risk_slope)
        * (components["risk"] - float(risk_threshold))
    )
    stability = stability_min + (
        stability_max - stability_min
    ) * stability_ratio
    remaining = 1.0 - stability
    task = task_energy_ratio * remaining
    energy = (1.0 - task_energy_ratio) * remaining
    weights = torch.stack((task, stability, energy), dim=-1)
    components["weights"] = weights
    return components


def weight_entropy(weights):
    return -(weights * (weights + 1e-8).log()).sum(dim=-1)


def payload_prediction_errors(prediction, target):
    """Return total and per-physical-block payload MSE diagnostics."""
    if prediction.shape != target.shape or prediction.shape[-1] != 9:
        raise ValueError(
            "payload prediction and target must both have shape (..., 9), "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    errors = (prediction - target).pow(2)
    return {
        "total": errors.mean(),
        "position": errors[..., 0:3].mean(),
        "torque": errors[..., 3:6].mean(),
        "velocity": errors[..., 6:9].mean(),
    }


def kl_prior_to_prediction(prior, prediction):
    prior = prior.clamp_min(1e-8)
    prediction = prediction.clamp_min(1e-8)
    return (prior * (prior.log() - prediction.log())).sum(dim=-1).mean()
