import torch

from .go2_env import Go2Env


class Go2TerrainMoeEnv(Go2Env):
    """Go2 environment exposing prediction targets for the terrain MoE method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_targets = None

    def step(self, actions):
        result = super().step(actions)

        next_dynamics = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos)
                * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
            ),
            dim=-1,
        )
        terrain_heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1)
            - 0.5
            - self.measured_heights,
            -1.0,
            1.0,
        ) * self.obs_scales.height_measurements
        self.moe_targets = {
            "dynamics": next_dynamics.detach(),
            "contact": self.contact_mask.float().detach(),
            "terrain": terrain_heights.detach(),
        }
        return result
