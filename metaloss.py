"""
MetaLoss: soft cross-entropy with <meta>...</meta> span masking and
object-term up-weighting (plus the sibling soft_cross_entropy helper from
the same source file).
"""

from typing import List, Union

import torch
from torch.nn.functional import cross_entropy
import torch.nn.functional as F

IGNORE_INDEX = -100  # was: from llava.constants import IGNORE_INDEX

__all__ = ["metaloss"]


def metaloss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    soft_tokens: Union[torch.Tensor, List[int]],
    tokenizer=None,
    object_terms: Union[List[List[str]], List[str], None] = None,
    object_weight: float = 3.0,
    ignore_index: int = -100,
    std: float = 1.0,
    meta_start_token: str = "<meta>",
    meta_end_token: str = "</meta>",
) -> torch.Tensor:
    """
    Soft cross entropy with masking (<meta>...</meta>), ignore_index masking,
    and optional object-term upweighting.
    """
    dtype = outputs.dtype
    # ---- Shift tokens ----
    outputs = outputs[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()

    batch_size, seq_len, vocab_size = outputs.shape

    # ---- Flatten ----
    flat_targets = targets.view(-1)
    flat_outputs = outputs.view(flat_targets.size(0), vocab_size)

    # ---- Base mask ----
    loss_mask = torch.ones_like(flat_targets, dtype=torch.float)

    # ---- Mask ignore_index ----
    ignore = flat_targets.eq(ignore_index)
    loss_mask = loss_mask.masked_fill(ignore, 0.0)

    # ---- Mask <meta> ... </meta> spans ----
    if tokenizer is not None:
        meta_start_id = tokenizer.convert_tokens_to_ids(meta_start_token)
        meta_end_id = tokenizer.convert_tokens_to_ids(meta_end_token)

        inside_meta = torch.zeros_like(targets, dtype=torch.bool)
        for b in range(batch_size):
            start_positions = (targets[b] == meta_start_id).nonzero(as_tuple=True)[0]
            end_positions = (targets[b] == meta_end_id).nonzero(as_tuple=True)[0]
            for s, e in zip(start_positions, end_positions):
                inside_meta[b, s:e + 1] = True

        # Flatten and mask out
        inside_meta = inside_meta.view(-1)
        loss_mask.masked_fill_(inside_meta, 0.0)

    # ---- Object term weighting ----
    if object_terms and tokenizer is not None:
        # normalize object_terms to batch-level list
        if isinstance(object_terms[0], str):
            object_terms = [object_terms] * batch_size

        assert len(object_terms) == batch_size, "object_terms batch size mismatch"

        for b in range(batch_size):
            terms_b = [t for t in object_terms[b] if isinstance(t, str) and len(t)]
            if not terms_b:
                continue
            obj_ids_b = set()
            for term in terms_b:
                ids = tokenizer.encode(term, add_special_tokens=False)
                obj_ids_b.update(ids)
            if obj_ids_b:
                # mask for this sample
                mask_b = torch.zeros(seq_len, dtype=torch.bool, device=targets.device)
                for oid in obj_ids_b:
                    mask_b |= (targets[b] == oid)
                # ignore already masked positions
                mask_b &= (targets[b] != ignore_index)
                mask_b &= (~inside_meta.view(batch_size, seq_len)[b])
                # upweight object token positions
                flat_idx = torch.arange(b * seq_len, (b + 1) * seq_len, device=targets.device)
                loss_mask[flat_idx[mask_b]] = object_weight

    # ---- Filter valid positions ----
    valid_indices = loss_mask > 0
    if valid_indices.sum() == 0:
        return outputs.new_tensor(0.0, requires_grad=True)

    outputs = flat_outputs[valid_indices]
    targets = flat_targets[valid_indices]
    weights = loss_mask[valid_indices]

    # ---- Soft token preparation ----
    if isinstance(soft_tokens, list):
        soft_tokens = torch.tensor(soft_tokens, device=targets.device)

    # ---- Build hard and soft masks (same shape as targets) ----
    is_soft = torch.isin(targets, soft_tokens)
    is_hard = ~is_soft

    # ---- Compute losses ----
    losses = torch.zeros_like(targets, dtype=dtype)

    # Hard loss
    if is_hard.any():
        hard_loss = F.cross_entropy(outputs[is_hard].float(), targets[is_hard], reduction="none").to(dtype)
        losses[is_hard] = hard_loss

    # Soft loss
    if is_soft.any():
        soft_out = outputs[is_soft]
        soft_tgt = targets[is_soft]
        soft_targets = torch.zeros_like(soft_out, dtype=dtype)
        for k, target in enumerate(soft_tgt):
            dist = torch.exp(-((target - soft_tokens) ** 2) / (2 * std**2))
            soft_targets[k][soft_tokens] = dist / dist.sum()
        soft_loss = F.cross_entropy(soft_out.float(), soft_targets.float(), reduction="none").to(dtype)
        losses[is_soft] = soft_loss

    # Weighted average
    total_loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
    return total_loss

