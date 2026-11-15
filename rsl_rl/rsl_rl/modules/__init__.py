# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .him_actor_critic import HIMActorCritic
from .him_estimator import HIMEstimator
from .actor_critic_dream import ActorCritic_dream
from .actor_critic_attention import ActorCritic_attention
from .actor_critic_attention1 import ActorCritic_attention1
from .normalizer import EmpiricalDiscountedVariationNormalization, EmpiricalHistoryNormalization,EmpiricalNormalization
from .actor_mulcritic_attention import Actor_mulCritic_attention1
from .actor_critic_servo import ActorCriticArmServo
from .actor_critic_film import ActorCriticArmServoDogPrior
from .actor_critic_no_film import ActorCriticArmServoDogPrior_nofilm
from .base_residual import BaseResidual
from .terrain_moe_actor_critic import TerrainMoeActorCritic
from .terrain_moe_attention import TerrainMoeAttentionActorCritic
from .terrain_actor_moe import TerrainActorMoEActorCritic
from .load_aware_weights import (
    LoadAwareWeightNetwork,
    physical_prior_weights,
    payload_prediction_errors,
    weight_entropy,
)
