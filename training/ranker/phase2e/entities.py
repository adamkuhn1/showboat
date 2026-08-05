"""Phase 2E: entity ("Deep Sets") representation of a Showboat shot decision.

THE KEY DESIGN CONSTRAINT, stated up front: the live game's runtime contract is
a single `[N, 68] float32 -> [N, 1] logit` ONNX graph (see
`docs/repair/showboat-ml/phase-2d/ARTIFACT_AND_PARITY.md`). Every entity feature
below is a *deterministic tensor function of those same 68 numbers* plus table
constants, computed INSIDE this module — which is itself an `nn.Module` made of
ONNX-exportable ops. So a relational model can be adopted with **zero change to
Team A's input contract**: same 68 floats in, same one logit out; the reshape
into entities happens inside the exported graph.

The one exception is `pot_id` (see `use_pot_id` below), which genuinely is not
recoverable from the 68 dims and therefore *would* require a contract change.
That variant is measured separately and flagged, not silently adopted.

## Where the 68 dims come from (verified against src/ai/ranker/encode.ts and
## src/ai/onnx.ts::encodeObservation, not guessed)

  dims  0..47 : board block = 16 balls x [x/halfLen, y/halfWid, pocketed]
                ball id is the slot index: 0 = cue, 1-7 solids, 8 = eight,
                9-15 stripes. A pocketed ball writes (0, 0, 1).
  dims 48..67 : candidate block, in encode.ts order:
                48 sin(phi)      49 cos(phi)     50 power
                51 sideSpin      52 topSpin      53..57 kind one-hot
                58 banks/4       59 target/15    60..65 pocket one-hot
                66 pathLength/diag               67 clearance/8

## Coordinate system used here ("u" units)

Board x and y are normalized by DIFFERENT constants (halfLen vs halfWid), so
raw normalized coordinates are anisotropic and any distance or angle computed
on them directly would be wrong. Everything below works in units of halfWid:
    x_u = x_norm * (halfLen/halfWid) = x_norm * 2   (regulation 2:1 table)
    y_u = y_norm
so the table is x_u in [-2, 2], y_u in [-1, 1], one ball radius = 0.0577 u, and
Euclidean distances and angles are physically meaningful.

## Entity features (per ball, ENTITY_DIM total)

identity: is_cue, is_solid, is_stripe, is_eight, is_target (first contact),
          is_pot (intended potted ball), pocketed.
          NOTE the deliberate omission of the ball's own number. Group identity
          (cue / solid / stripe / eight) is what 8-ball rules and physics care
          about; *which* stripe a ball is carries no physical information and
          including it would let the model latch onto ball indices. Omitting it
          buys a real, testable symmetry: swapping the positions of two
          same-group balls that are neither the target nor the intended pot
          must leave the prediction bit-identical. `controls.py` tests exactly
          that, and a flat MLP over the same features fails it.
position: x, y (masked to 0 when pocketed), distance to nearest cushion
relations to cue: dx, dy, distance, and the cosine/sine of the angle between
          (ball - cue) and the aim direction phi
relations to the target ball and to the intended pocket: dx, dy, distance each
path relations (the actual point of a relational model — "is this ball in the
          way of *this specific shot*"):
          - cue -> target segment: along-track fraction, perpendicular
            distance, in-segment flag, smooth blocking score
          - intended-pot-ball -> pocket segment: same four
          - the aim ray from the cue along phi: along-track, perpendicular,
            smooth blocking score (this is the true first leg of the shot even
            for bank shots, where cue->target is NOT the real path)
          - bank geometry: the standard mirror construction. For each of the 4
            cushions, reflect the intended pocket across it and score every
            ball against the pot-ball -> mirrored-pocket segment (object-ball
            bank), and reflect the target ball across it and score against the
            cue -> mirrored-target segment (cue-ball bank). 4 cushions x 2
            constructions x (perp, block) = 16 dims. This is the only place the
            model gets any information about multi-cushion path geometry, since
            the 68-dim encoding compresses the whole real path down to two
            scalars (length, clearance).

`RELATION_SLICE` marks the contiguous block of path/bank relation dims so the
relationship-feature ablation can zero exactly those and nothing else.
"""

from __future__ import annotations

import torch
from torch import nn

# --- table constants, mirrored from src/physics/constants.ts -----------------
TABLE_LENGTH = 1.9812
TABLE_WIDTH = 0.9906
BALL_RADIUS = 0.028575
HALF_LEN = TABLE_LENGTH / 2
HALF_WID = TABLE_WIDTH / 2
AX = HALF_LEN / HALF_WID  # 2.0 on a regulation 2:1 table
R_U = BALL_RADIUS / HALF_WID  # ball radius in u units (~0.0577)

N_BALLS = 16
BOARD_DIM = 48
TOTAL_DIM = 68

# pocket centres in u units, in encode.ts's POCKET_IDS order (bl,tl,br,tr,sb,st)
POCKET_XY = torch.tensor(
    [[-AX, -1.0], [-AX, 1.0], [AX, -1.0], [AX, 1.0], [0.0, -1.0], [0.0, 1.0]],
    dtype=torch.float32,
)

# candidate-block absolute indices
I_SIN_PHI, I_COS_PHI = 48, 49
I_KIND0 = 53
I_TARGET_NORM = 59
I_POCKET0 = 60

EPS = 1e-8

# --- entity feature layout ---------------------------------------------------
_ID_DIMS = 7       # is_cue,is_solid,is_stripe,is_eight,is_target,is_pot,pocketed
_POS_DIMS = 3      # x, y, cushion distance
_CUE_DIMS = 5      # dx, dy, dist, cos/sin vs aim
_TGT_DIMS = 3
_POCK_DIMS = 3
_REL_DIMS = 4 + 4 + 3 + 16  # segA, segB, aim ray, bank mirrors
ENTITY_DIM = _ID_DIMS + _POS_DIMS + _CUE_DIMS + _TGT_DIMS + _POCK_DIMS + _REL_DIMS
RELATION_OFFSET = _ID_DIMS + _POS_DIMS + _CUE_DIMS + _TGT_DIMS + _POCK_DIMS
RELATION_SLICE = slice(RELATION_OFFSET, ENTITY_DIM)

CONTEXT_DIM = 20 + 1 + 6 + 3 + 2 + 2 + 1  # see the context block in forward()


def _seg_rel(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
    """Relation of every entity position P [N,16,2] to the segment A->B [N,2].

    Returns (t_frac, perp, in_segment, block), each [N,16]:
      t_frac     along-track position as a fraction of segment length
      perp       perpendicular distance in u units
      in_segment 1 when the foot of the perpendicular lies inside the segment
      block      in_segment * exp(-perp / ball diameter) — a smooth "this ball
                 obstructs this segment" score; ~1 when the ball sits on the
                 line, decaying over roughly one ball diameter
    """
    d = B - A                                             # [N,2]
    L = torch.sqrt((d * d).sum(-1, keepdim=True) + EPS)   # [N,1]
    dn = d / L
    v = P - A.unsqueeze(1)                                # [N,16,2]
    t = (v * dn.unsqueeze(1)).sum(-1)                     # [N,16]
    proj = t.unsqueeze(-1) * dn.unsqueeze(1)
    perp = torch.sqrt(((v - proj) ** 2).sum(-1) + EPS)
    Lb = L.squeeze(-1)
    in_seg = (t > 0).float() * (t < Lb.unsqueeze(1)).float()
    block = in_seg * torch.exp(-perp / (2.0 * R_U))
    # Clamped: when the segment is near-degenerate (cue almost touching the
    # target ball, or the pot ball sitting in the jaws) the along-track
    # *fraction* diverges — measured up to 23 before clamping. The in/out and
    # perpendicular features already carry the information; an unbounded
    # feature just destabilises training.
    t_frac = torch.clamp(t / (Lb.unsqueeze(1) + EPS), -4.0, 4.0)
    return t_frac, perp, in_seg, block


def _ray_rel(P: torch.Tensor, A: torch.Tensor, dirv: torch.Tensor):
    """Same, for an infinite ray from A along unit direction `dirv` [N,2]."""
    v = P - A.unsqueeze(1)
    t = (v * dirv.unsqueeze(1)).sum(-1)
    proj = t.unsqueeze(-1) * dirv.unsqueeze(1)
    perp = torch.sqrt(((v - proj) ** 2).sum(-1) + EPS)
    front = (t > 0).float()
    block = front * torch.exp(-perp / (2.0 * R_U))
    return t / (2.0 * AX), perp, block


def _mirror(pt: torch.Tensor, cushion: int) -> torch.Tensor:
    """Reflect a point [N,2] (u units) across cushion 0..3 = x=-AX, x=+AX,
    y=-1, y=+1 — the textbook bank-shot aiming construction."""
    x, y = pt[:, 0], pt[:, 1]
    if cushion == 0:
        return torch.stack([-2.0 * AX - x, y], dim=-1)
    if cushion == 1:
        return torch.stack([2.0 * AX - x, y], dim=-1)
    if cushion == 2:
        return torch.stack([x, -2.0 - y], dim=-1)
    return torch.stack([x, 2.0 - y], dim=-1)


class EntityEncoder(nn.Module):
    """[N,68] (+ optional pot id column) -> (entities [N,16,ENTITY_DIM],
    mask [N,16], context [N,CONTEXT_DIM]).

    Pure tensor arithmetic, no learned parameters, no data-dependent control
    flow — so it exports to ONNX as part of the model graph and the browser
    keeps feeding the exact same 68 floats it already produces.
    """

    def __init__(self, use_pot_id: bool = False, zero_relations: bool = False):
        super().__init__()
        self.use_pot_id = use_pot_id
        self.zero_relations = zero_relations
        self.register_buffer("pocket_xy", POCKET_XY.clone())
        self.register_buffer("ball_ids", torch.arange(N_BALLS, dtype=torch.float32))
        ident = torch.zeros(N_BALLS, 4)
        ident[0, 0] = 1.0                       # cue
        ident[1:8, 1] = 1.0                     # solids
        ident[9:16, 2] = 1.0                    # stripes
        ident[8, 3] = 1.0                       # eight
        self.register_buffer("ident", ident)

    def forward(self, x: torch.Tensor):
        # `-1` rather than `x.shape[0]` everywhere below: the ONNX tracer would
        # otherwise bake the export-time batch size in as a constant, and the
        # live contract needs a dynamic batch axis (one call scores all of a
        # turn's candidates at once).
        board = x[:, :BOARD_DIM].reshape(-1, N_BALLS, 3)
        pos = torch.stack([board[:, :, 0] * AX, board[:, :, 1]], dim=-1)  # [N,16,2]
        pocketed = board[:, :, 2]
        live = 1.0 - pocketed
        pos = pos * live.unsqueeze(-1)

        cand = x[:, BOARD_DIM:TOTAL_DIM]

        # --- which ball is first contact, which is the intended pot
        tgt_id = x[:, I_TARGET_NORM : I_TARGET_NORM + 1] * (N_BALLS - 1)   # [N,1]
        is_target = (torch.abs(tgt_id - self.ball_ids.unsqueeze(0)) < 0.5).float()
        if self.use_pot_id:
            pot_id = x[:, TOTAL_DIM : TOTAL_DIM + 1] * (N_BALLS - 1)
            is_pot = (torch.abs(pot_id - self.ball_ids.unsqueeze(0)) < 0.5).float()
        else:
            is_pot = is_target

        cue = pos[:, 0, :]                                        # [N,2]
        tgt = torch.bmm(is_target.unsqueeze(1), pos).squeeze(1)   # [N,2]
        potb = torch.bmm(is_pot.unsqueeze(1), pos).squeeze(1)
        pocket = x[:, I_POCKET0 : I_POCKET0 + 6] @ self.pocket_xy  # [N,2]

        sin_phi = x[:, I_SIN_PHI]
        cos_phi = x[:, I_COS_PHI]
        aim = torch.stack([cos_phi, sin_phi], dim=-1)             # [N,2], unit by construction

        # --- identity block
        ident = self.ident.unsqueeze(0) * torch.ones_like(pocketed).unsqueeze(-1)
        idblock = torch.cat(
            [
                ident,
                is_target.unsqueeze(-1),
                is_pot.unsqueeze(-1),
                pocketed.unsqueeze(-1),
            ],
            dim=-1,
        )

        # --- position block
        cush = torch.minimum(AX - torch.abs(pos[:, :, 0]), 1.0 - torch.abs(pos[:, :, 1]))
        posblock = torch.cat([pos, cush.unsqueeze(-1)], dim=-1)

        # --- relation to the cue ball, oriented by the aim direction
        dcue = pos - cue.unsqueeze(1)
        dist_cue = torch.sqrt((dcue * dcue).sum(-1) + EPS)
        ucue = dcue / dist_cue.unsqueeze(-1)
        cos_aim = (ucue * aim.unsqueeze(1)).sum(-1)
        sin_aim = ucue[:, :, 0] * aim[:, 1].unsqueeze(1) - ucue[:, :, 1] * aim[:, 0].unsqueeze(1)
        cueblock = torch.cat(
            [dcue, dist_cue.unsqueeze(-1), cos_aim.unsqueeze(-1), sin_aim.unsqueeze(-1)],
            dim=-1,
        )

        dtgt = pos - tgt.unsqueeze(1)
        dist_tgt = torch.sqrt((dtgt * dtgt).sum(-1) + EPS)
        tgtblock = torch.cat([dtgt, dist_tgt.unsqueeze(-1)], dim=-1)

        dpk = pos - pocket.unsqueeze(1)
        dist_pk = torch.sqrt((dpk * dpk).sum(-1) + EPS)
        pkblock = torch.cat([dpk, dist_pk.unsqueeze(-1)], dim=-1)

        # --- path relations
        tA, pA, iA, bA = _seg_rel(pos, cue, tgt)
        tB, pB, iB, bB = _seg_rel(pos, potb, pocket)
        tR, pR, bR = _ray_rel(pos, cue, aim)
        rel = [
            tA.unsqueeze(-1), pA.unsqueeze(-1), iA.unsqueeze(-1), bA.unsqueeze(-1),
            tB.unsqueeze(-1), pB.unsqueeze(-1), iB.unsqueeze(-1), bB.unsqueeze(-1),
            tR.unsqueeze(-1), pR.unsqueeze(-1), bR.unsqueeze(-1),
        ]
        for c in range(4):
            _, pm, _, bm = _seg_rel(pos, potb, _mirror(pocket, c))   # object-ball bank
            rel += [pm.unsqueeze(-1), bm.unsqueeze(-1)]
        for c in range(4):
            _, pm, _, bm = _seg_rel(pos, cue, _mirror(tgt, c))       # cue-ball bank
            rel += [pm.unsqueeze(-1), bm.unsqueeze(-1)]
        relblock = torch.cat(rel, dim=-1)
        if self.zero_relations:
            relblock = relblock * 0.0

        ent = torch.cat([idblock, posblock, cueblock, tgtblock, pkblock, relblock], dim=-1)

        # --- global context
        n_live = live.sum(-1, keepdim=True) / (N_BALLS - 1)
        d_ct = torch.sqrt(((tgt - cue) ** 2).sum(-1, keepdim=True) + EPS)
        d_tp = torch.sqrt(((pocket - potb) ** 2).sum(-1, keepdim=True) + EPS)
        d_cp = torch.sqrt(((pocket - cue) ** 2).sum(-1, keepdim=True) + EPS)
        u1 = (tgt - cue) / d_ct
        u2 = (pocket - potb) / d_tp
        cut_cos = (u1 * u2).sum(-1, keepdim=True)
        cut_sin = u1[:, 0:1] * u2[:, 1:2] - u1[:, 1:2] * u2[:, 0:1]
        cush_t = torch.minimum(AX - torch.abs(tgt[:, 0:1]), 1.0 - torch.abs(tgt[:, 1:2]))
        cush_c = torch.minimum(AX - torch.abs(cue[:, 0:1]), 1.0 - torch.abs(cue[:, 1:2]))
        is_combo = 1.0 - (is_target * is_pot).sum(-1, keepdim=True)
        ctx = torch.cat(
            [cand, n_live, cue, tgt, potb, d_ct, d_tp, d_cp,
             cut_cos, cut_sin, cush_t, cush_c, is_combo],
            dim=-1,
        )
        return ent, live, ctx
