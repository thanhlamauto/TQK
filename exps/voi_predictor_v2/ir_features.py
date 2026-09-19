"""Frozen ImageReward hidden features (no fine-tuning).

``IRSMC.score_batched_tensor`` computes a 768-d fused BLIP representation
(``text_output.last_hidden_state[:, 0, :]``) before the reward MLP.  This module
replicates that forward and returns the representation, reusing the already
loaded reward model.
"""

from __future__ import annotations

import time

import torch


def extract_ir_hidden(model, prompts, image_tensors, *, from_minus_one_to_one: bool = True,
                      emulate_pil_uint8_roundtrip: bool = True) -> torch.Tensor:
    """Return (n, 768) frozen ImageReward hidden features for the given images."""
    assert isinstance(prompts, list)
    assert isinstance(image_tensors, torch.Tensor)
    text_input = model.blip.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    ).to(model.device)
    images = model._prepare_tensor_images(
        image_tensors,
        from_minus_one_to_one=from_minus_one_to_one,
        emulate_pil_uint8_roundtrip=emulate_pil_uint8_roundtrip,
    )
    image_embeds = model.blip.visual_encoder(images)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(model.device)
    text_output = model.blip.text_encoder(
        text_input.input_ids,
        attention_mask=text_input.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    hidden = text_output.last_hidden_state[:, 0, :].float()
    del images, image_embeds, text_output
    return hidden
