#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.eef_action_processor import (
    EEFActionProcessorStep,
    EEFUnnormalizeProcessorStep,
)
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_diffusion import DiffusionConfig


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.

    Pre-processing order:
    1. Rename features.
    2. Add batch dimension.
    3. Move to device.
    4. (Optional) Convert absolute actions → relative deltas.
    5. Normalize using dataset statistics.

    Post-processing order:
    1. Unnormalize predictions.
    2. (Optional) Convert relative deltas → absolute actions by adding cached state.
    3. Move to CPU.
    """
    if config.use_eef_actions:
        if not config.eef_poses_path or not config.eef_stats_path:
            raise ValueError(
                "use_eef_actions=True requires eef_poses_path and eef_stats_path to be set."
            )
        action_step = EEFActionProcessorStep(
            eef_poses_path=config.eef_poses_path,
            eef_stats_path=config.eef_stats_path,
            horizon=config.horizon,
        )
        post_action_step = EEFUnnormalizeProcessorStep(eef_stats_path=config.eef_stats_path)
    else:
        action_step = RelativeActionsProcessorStep(
            enabled=config.use_relative_actions,
            exclude_joints=getattr(config, "relative_exclude_joints", []),
            action_names=getattr(config, "action_feature_names", None),
        )
        post_action_step = AbsoluteActionsProcessorStep(
            enabled=config.use_relative_actions, relative_step=action_step
        )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        action_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        post_action_step,
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
