from __future__ import annotations

import torch
import torch.nn.functional as F


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x[indices]


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices, hidden_states)
    return output.view(batch, seqlen, *hidden_states.shape[1:])


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return index_first_axis(hidden_states.reshape(-1, *hidden_states.shape[2:]), indices), indices, cu_seqlens, max_seqlen
