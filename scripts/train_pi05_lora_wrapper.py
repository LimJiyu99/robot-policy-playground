#!/usr/bin/env python3
"""Inject the supported PEFT LoraConfig that LeRobot's CLI cannot fully express.

The installed CLI exposes PEFT target/rank/alpha but not lora_dropout.  This
wrapper leaves LeRobot's training implementation unchanged and only replaces
the policy's PEFT factory call with an official ``peft.LoraConfig``.
"""

from __future__ import annotations

import logging

from torch import nn
from peft import LoraConfig

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.scripts import lerobot_train


ACTION_EXPERT_MODULES = [
    "model.paligemma_with_expert.gemma_expert",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.time_mlp_in",
    "model.time_mlp_out",
]
_ORIGINAL_WRAP_WITH_PEFT = PreTrainedPolicy.wrap_with_peft
LOGGER = logging.getLogger(__name__)


def get_vlm_language_attention_targets(policy: PreTrainedPolicy) -> list[str]:
    """Return existing VLM language-attention Linear module names only."""
    prefix = "model.paligemma_with_expert.paligemma.model.language_model.layers."
    projections = {"q_proj", "k_proj", "v_proj", "o_proj"}
    return [
        name
        for name, module in policy.named_modules()
        if isinstance(module, nn.Linear)
        and name.startswith(prefix)
        and ".self_attn." in name
        and name.rsplit(".", 1)[-1] in projections
    ]


def wrap_pi05_with_lora(
    self: PreTrainedPolicy,
    peft_config=None,
    peft_cli_overrides: dict | None = None,
):
    """Train VLM-language attention LoRA plus complete Action Expert modules."""
    if self.name != "pi05":
        raise ValueError(f"This wrapper only supports pi05, got {self.name!r}")
    target_modules = get_vlm_language_attention_targets(self)
    LOGGER.info(
        "PI05 VLM language-attention LoRA targets: count=%d first_20=%s",
        len(target_modules),
        target_modules[:20],
    )
    if not target_modules:
        raise RuntimeError("No VLM language-attention Linear modules found; refusing to start training.")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        modules_to_save=ACTION_EXPERT_MODULES,
        bias="none",
    )
    # ``--peft.method_type`` only activates LeRobot's PEFT path; preserve this
    # explicit config rather than letting the reduced CLI dataclass overwrite
    # its dropout/target values.
    peft_model = _ORIGINAL_WRAP_WITH_PEFT(self, peft_config=config, peft_cli_overrides=None)
    adapted_modules = [name for name, module in peft_model.named_modules() if hasattr(module, "lora_A")]
    LOGGER.info(
        "PI05 LoRA applied modules: count=%d first_20=%s",
        len(adapted_modules),
        adapted_modules[:20],
    )
    if len(adapted_modules) != len(target_modules):
        raise RuntimeError(
            f"Expected {len(target_modules)} LoRA modules but found {len(adapted_modules)}; refusing to train."
        )
    return peft_model


PreTrainedPolicy.wrap_with_peft = wrap_pi05_with_lora
lerobot_train.main()
