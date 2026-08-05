"""Phase 2E architectures.

Four models, all consuming the identical `[N, 68]` row the live game already
produces (the pot-id variant appends one extra column and is flagged as a
contract change):

  TinyNet          - Phase 2D's frozen candidate-conditioned MLP. Reproduced
                     here bit-identically (same layer sizes, same init order)
                     so its checkpoints load and so a matched-protocol retrain
                     is possible; NOT re-invented.
  FlatEntityMLP    - the same derived entity features as the relational model,
                     but flattened into one 16*ENTITY_DIM + CONTEXT_DIM vector
                     and fed to a plain MLP. This exists to separate two things
                     that are easy to confuse: how much of any gain comes from
                     *better features* vs. how much comes from the
                     *permutation-invariant architecture*.
  DeepSetsRanker   - candidate-conditioned Deep Sets. Each ball is an entity;
                     entity features are concatenated with the shot context,
                     passed through a shared phi MLP, pooled over live balls
                     with masked mean + masked max (both permutation
                     invariant), then decoded with rho together with the
                     context.
  DeepSetsAttn     - same, with one masked multi-head self-attention layer over
                     the entity set before pooling (i.e. a set transformer /
                     fully-connected relational GNN layer over 16 nodes). This
                     is the "GNN" arm: on a 16-node fully-connected graph a
                     message-passing GNN and a masked self-attention block are
                     the same computation, so this is implemented as attention
                     rather than pulling in a graph library that would not
                     export to ONNX cleanly.
"""

from __future__ import annotations

import torch
from torch import nn

from entities import CONTEXT_DIM, ENTITY_DIM, EntityEncoder


class TinyNet(nn.Module):
    """Phase 2D's frozen baseline: 68 -> 32 -> 32 -> 1, ReLU.

    Kept byte-compatible with `phase2c/diagnostic.py::TinyNet` (same module
    names, so its saved `state_dict`s load directly)."""

    def __init__(self, input_dim: int = 68, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FlatEntityMLP(nn.Module):
    """Ablation arm: same entity features, no permutation invariance."""

    def __init__(self, hidden: int = 64, use_pot_id: bool = False, zero_relations: bool = False):
        super().__init__()
        self.enc = EntityEncoder(use_pot_id=use_pot_id, zero_relations=zero_relations)
        self.net = _mlp([16 * ENTITY_DIM + CONTEXT_DIM, hidden, hidden, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ent, mask, ctx = self.enc(x)
        ent = ent * mask.unsqueeze(-1)
        flat = torch.cat([ent.reshape(ent.shape[0], -1), ctx], dim=-1)
        return self.net(flat).squeeze(-1)


class DeepSetsRanker(nn.Module):
    """Candidate-conditioned Deep Sets over the 16 balls.

    Permutation invariance comes from the pooling step and nothing else: phi is
    applied identically to every entity, and masked mean/max are symmetric
    functions of the set. There is no positional embedding and no ball-number
    feature anywhere, so the invariance is real rather than approximate — see
    `controls.py::permutation_invariance`.

    Masking: pocketed balls are excluded from both pools (they carry no
    position). If every ball were pocketed the mean would divide by zero, so
    the denominator is clamped at 1 — an impossible state in practice (the cue
    is always live) but a guard that keeps the exported graph total.
    """

    def __init__(
        self,
        hidden: int = 64,
        phi_hidden: int = 64,
        use_pot_id: bool = False,
        zero_relations: bool = False,
    ):
        super().__init__()
        self.enc = EntityEncoder(use_pot_id=use_pot_id, zero_relations=zero_relations)
        self.phi = _mlp([ENTITY_DIM + CONTEXT_DIM, phi_hidden, phi_hidden])
        self.rho = _mlp([2 * phi_hidden + CONTEXT_DIM, hidden, hidden, 1])

    def pool(self, ent: torch.Tensor, mask: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        c = ctx.unsqueeze(1).expand(-1, ent.shape[1], -1)
        h = self.phi(torch.cat([ent, c], dim=-1))          # [N,16,H]
        m = mask.unsqueeze(-1)
        s = (h * m).sum(1)
        n = mask.sum(1, keepdim=True).clamp(min=1.0)
        mean = s / n
        mx = (h * m + (m - 1.0) * 1e4).max(dim=1).values    # masked max
        return torch.cat([mean, mx, ctx], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ent, mask, ctx = self.enc(x)
        return self.rho(self.pool(ent, mask, ctx)).squeeze(-1)

    def forward_from_entities(self, ent, mask, ctx) -> torch.Tensor:
        """Bypass the encoder — used by the permutation-invariance control to
        feed a deliberately reordered entity set."""
        return self.rho(self.pool(ent, mask, ctx)).squeeze(-1)


class DeepSetsAttn(nn.Module):
    """Deep Sets + one masked multi-head self-attention layer ("GNN" arm).

    On a 16-node fully-connected graph, a message-passing layer with learned
    edge weights and a masked self-attention block compute the same family of
    functions. Attention is used because it is a handful of standard ops that
    ONNX exports natively, whereas a graph library's scatter/gather kernels do
    not. Still permutation-equivariant before pooling and invariant after.
    """

    def __init__(
        self,
        hidden: int = 64,
        phi_hidden: int = 64,
        heads: int = 4,
        use_pot_id: bool = False,
        zero_relations: bool = False,
    ):
        super().__init__()
        self.enc = EntityEncoder(use_pot_id=use_pot_id, zero_relations=zero_relations)
        self.phi = _mlp([ENTITY_DIM + CONTEXT_DIM, phi_hidden, phi_hidden])
        self.attn = nn.MultiheadAttention(phi_hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(phi_hidden)
        self.rho = _mlp([2 * phi_hidden + CONTEXT_DIM, hidden, hidden, 1])

    def pool(self, ent, mask, ctx):
        c = ctx.unsqueeze(1).expand(-1, ent.shape[1], -1)
        h = self.phi(torch.cat([ent, c], dim=-1))
        pad = mask < 0.5
        # A row with every entity masked would produce NaNs in softmax; the cue
        # is always live so this cannot happen with real data, but keep the
        # graph total by forcing at least one visible key.
        pad = pad & ~(pad.all(dim=1, keepdim=True))
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        h = self.norm(h + a)
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        mx = (h * m + (m - 1.0) * 1e4).max(dim=1).values
        return torch.cat([mean, mx, ctx], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ent, mask, ctx = self.enc(x)
        return self.rho(self.pool(ent, mask, ctx)).squeeze(-1)

    def forward_from_entities(self, ent, mask, ctx) -> torch.Tensor:
        return self.rho(self.pool(ent, mask, ctx)).squeeze(-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
