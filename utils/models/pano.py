# Copyright 2026 Zijiang Yang.
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

import math
import torch
from timm.layers import trunc_normal_
from .op import faster_sinkhorn

ACTIVATION = {
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ReLU": torch.nn.ReLU(),
    "LeakyReLU": torch.nn.LeakyReLU(0.1),
    "ELU": torch.nn.ELU(),
    "GELU": torch.nn.GELU(),
}


def timestep_embedding(timesteps, dim, max_period=10000):
    """Build sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PANOMLP(torch.nn.Module):
    """Residual MLP used by feature, routing, and output projectors."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act, drop_rate=0.0):
        super().__init__()
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)]
        )
        self.output = torch.nn.Linear(hidden_dim, output_dim)
        self.act = act
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, x):
        hidden = self.drop(self.act(self.input(x)))
        for layer in self.hidden:
            hidden = hidden + self.drop(self.act(layer(hidden)))
        return self.output(hidden)


class TimeProjector(torch.nn.Module):
    """Project a scalar timestep and broadcast it to every input point."""

    def __init__(self, n_dim, n_layer, act):
        super().__init__()
        self.projector = PANOMLP(n_dim, n_dim, n_dim, n_layer, act)
        self.n_dim = n_dim

    def forward(self, t, n_point):
        if t.dim() > 1:
            t = t.reshape(t.shape[0])
        time_feature = timestep_embedding(t, self.n_dim)
        time_feature = self.projector(time_feature)
        return time_feature.unsqueeze(1).expand(-1, n_point, -1)


class PANOSelfAttention(torch.nn.Module):
    """
    Explicit multi-head self-attention used by PANO.

    This is intentionally the non-fused attention path of the reference model.
    """

    def __init__(self, n_dim, n_head, drop_rate=0.0):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_dim // n_head) ** (-0.5)
        self.fused_attn = False
        self.attn_drop = torch.nn.Dropout(drop_rate)
        self.Wq = torch.nn.Linear(n_dim, n_dim)
        self.Wk = torch.nn.Linear(n_dim, n_dim)
        self.Wv = torch.nn.Linear(n_dim, n_dim)
        self.proj = torch.nn.Linear(n_dim, n_dim)

    def forward(self, x):
        batch, tokens, dim = x.size()
        q = (
            self.Wq(x)
            .view(batch, tokens, self.n_head, dim // self.n_head)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.Wk(x)
            .view(batch, tokens, self.n_head, dim // self.n_head)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.Wv(x)
            .view(batch, tokens, self.n_head, dim // self.n_head)
            .permute(0, 2, 1, 3)
        )
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, tokens, dim)
        return self.proj(out)


class PANOAttentionBlock(torch.nn.Module):
    """Pre-normalized self-attention block followed by a residual MLP."""

    def __init__(self, n_dim, n_head, act, drop_rate=0.0):
        super().__init__()
        self.self_attn = PANOSelfAttention(n_dim, n_head, drop_rate=drop_rate)
        self.ln1 = torch.nn.LayerNorm(n_dim)
        self.ln2 = torch.nn.LayerNorm(n_dim)
        self.drop = torch.nn.Dropout(drop_rate)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_dim, 4 * n_dim), act, torch.nn.Linear(4 * n_dim, n_dim)
        )

    def forward(self, y):
        y = y + self.drop(self.self_attn(self.ln1(y)))
        y = y + self.mlp(self.ln2(y))
        return y


class PaLA(torch.nn.Module):
    """Pre-norm pairwise interaction for independent [B, H, T, C] sequences.

    A shared matching projection gives S_ji = S_ij.T. Each direction
    normalizes over its own source tokens; updates use the same input state.
    H=1 is identity, including the FFN, since there is no interaction.
    """

    def __init__(self, n_dim, n_head, act, drop_rate=0.0):
        super().__init__()
        if n_head <= 0 or n_dim % n_head:
            raise ValueError("n_dim must be divisible by positive n_head.")
        self.n_head = n_head
        self.scale = (n_dim // n_head) ** (-0.5)
        self.ln1 = torch.nn.LayerNorm(n_dim)
        self.match_proj = torch.nn.Linear(n_dim, n_dim)
        self.value_proj = torch.nn.Linear(n_dim, n_dim)
        self.output_proj = torch.nn.Linear(n_dim, n_dim)
        self.attn_drop = torch.nn.Dropout(drop_rate)
        self.drop = torch.nn.Dropout(drop_rate)
        self.ln2 = torch.nn.LayerNorm(n_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_dim, 4 * n_dim), act, torch.nn.Linear(4 * n_dim, n_dim)
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B, H, T, C], got {tuple(x.shape)}.")
        batch, heads, tokens, dim = x.shape
        if heads == 1:
            return x
        z = self.ln1(x)
        match = (
            self.match_proj(z)
            .reshape(batch, heads, tokens, self.n_head, dim // self.n_head)
            .permute(0, 1, 3, 2, 4)
        )
        value = (
            self.value_proj(z)
            .reshape(batch, heads, tokens, self.n_head, dim // self.n_head)
            .permute(0, 1, 3, 2, 4)
        )
        messages = [torch.zeros_like(value[:, h]) for h in range(heads)]
        for i in range(heads):
            for j in range(i + 1, heads):
                score = match[:, i] @ match[:, j].transpose(-2, -1) * self.scale
                to_i = self.attn_drop(score.softmax(dim=-1)) @ value[:, j]
                to_j = (
                    self.attn_drop(score.transpose(-2, -1).softmax(dim=-1))
                    @ value[:, i]
                )
                messages[i] = messages[i] + to_i
                messages[j] = messages[j] + to_j
        update = torch.stack(messages, dim=1).permute(0, 1, 3, 2, 4).reshape(
            batch, heads, tokens, dim
        ) / (heads - 1)
        x = x + self.drop(self.output_proj(update))
        return x + self.mlp(self.ln2(x))


class PANOLatentStack(torch.nn.Module):
    """Stack multiple attention blocks in the mode/latent space."""

    def __init__(self, n_block, n_dim, n_head, act, drop_rate=0.0):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            *[
                PANOAttentionBlock(n_dim, n_head, act, drop_rate=drop_rate)
                for _ in range(n_block)
            ]
        )

    def forward(self, latent):
        return self.blocks(latent)


class PANOInitialProjector(torch.nn.Module):
    """Lift point conditions into features used for both routing and values.
    """

    def __init__(self, n_dim, n_layer, condition_dim, act):
        super().__init__()
        self.projector = PANOMLP(condition_dim, n_dim, n_dim, n_layer, act)

    def forward(self, y):
        return self.projector(y)


class MeanMultiheadDecodeMerge(torch.nn.Module):
    """Merge decoded [B, H, N, C] features by averaging heads."""

    def forward(self, x):
        return x.mean(dim=1)


class CAP(torch.nn.Module):
    """Independent latent-space routing with a single stored FP32 decode map.

    Encode uses E=(M/N)*D without materializing E. Routing MLPs follow
    autocast; Sinkhorn and point/latent transport compute in FP32. Transport
    restores the input feature dtype before subsequent attention blocks.
    """

    def __init__(self, n_dim, n_mode, n_layer, act, n_spaces=4, n_iter=1):
        super().__init__()
        self.n_dim = n_dim
        self.n_mode = n_mode
        self.n_iter = n_iter
        self.n_spaces = n_spaces
        projector_dim = self.projector_dim()
        self.attention_projectors = torch.nn.ModuleList(
            [
                PANOMLP(projector_dim, projector_dim, n_mode, n_layer, act)
                for _ in range(n_spaces)
            ]
        )
        self.out_dim = n_dim
        self.out_merge = MeanMultiheadDecodeMerge()

    def projector_dim(self):
        return self.n_dim

    def project_scores(self, score_source):
        return torch.stack(
            [projector(score_source) for projector in self.attention_projectors], dim=1
        )

    def build_mapping(self, score_source):
        score = self.project_scores(score_source)
        batch, heads, points, modes = score.shape
        with torch.autocast(device_type=score.device.type, enabled=False):
            decode = faster_sinkhorn(
                score.reshape(batch * heads, points, modes),
                n_iter=self.n_iter,
                temperature=1.0,
                inverse_update=False,
                decode_only=True,
            )
        return {"score_decode": decode.reshape(batch, heads, points, modes)}

    def encode_mapping(self, mapping, source_feature):
        return self.encode(mapping["score_decode"], source_feature)

    def encode(self, score_decode, source_feature):
        if score_decode.dtype != torch.float32:
            raise TypeError("CAP mapping must be FP32.")
        batch, heads, points, modes = score_decode.shape
        with torch.autocast(device_type=source_feature.device.type, enabled=False):
            latent = score_decode.transpose(-2, -1) @ source_feature.float().unsqueeze(
                1
            )
            latent = latent * (modes / points)
        return latent.reshape(batch * heads, modes, self.n_dim).to(source_feature.dtype)

    def decode(self, score_decode, latent_feature):
        if score_decode.dtype != torch.float32:
            raise TypeError("CAP mapping must be FP32.")
        batch, heads, _, modes = score_decode.shape
        if tuple(latent_feature.shape) != (batch * heads, modes, self.n_dim):
            raise ValueError("Latent feature shape does not match the decode mapping.")
        with torch.autocast(device_type=latent_feature.device.type, enabled=False):
            latent = latent_feature.float().reshape(batch, heads, modes, self.n_dim)
            decoded = self.out_merge(score_decode @ latent)
        return decoded.to(latent_feature.dtype)


class PANOBlock(torch.nn.Module):
    """One flat encode/latent-program/decode round trip.

    The ordered program allows repeated operation types, e.g.
    (("attnlayer", 3), ("pairwisecross", 1), ("attnlayer", 3)).
    Construction puts pairwise blocks first to preserve the reference RNG
    sequence. Execution always follows the declaration; weight reinitialization
    visits attention groups first, then pairwise groups.
    """

    def __init__(
        self,
        name,
        program,
        n_mode,
        n_dim,
        n_head,
        n_layer,
        act,
        drop_rate,
        n_spaces,
        n_iter,
    ):
        super().__init__()
        self.name = name
        self.program = tuple((tuple(item) for item in program))
        self.projector = CAP(
            n_dim, n_mode, n_layer, act, n_spaces=n_spaces, n_iter=n_iter
        )
        groups = {}
        for kind in ("pairwisecross", "attnlayer"):
            for index, (op, count) in enumerate(self.program):
                if op != kind:
                    continue
                if kind == "attnlayer":
                    groups[index] = PANOLatentStack(
                        count, n_dim, n_head, act, drop_rate
                    )
                else:
                    groups[index] = torch.nn.Sequential(
                        *[PaLA(n_dim, n_head, act, drop_rate) for _ in range(count)]
                    )
        self.operations = torch.nn.ModuleList(
            [groups[i] for i in range(len(self.program))]
        )

    def initialization_stacks(self):
        for kind in ("attnlayer", "pairwisecross"):
            for (op, _), stack in zip(self.program, self.operations):
                if op == kind:
                    yield stack

    def forward(self, source_feature):
        """Route point features through independent latent spaces."""
        mapping = self.projector.build_mapping(source_feature)
        latent = self.projector.encode_mapping(mapping, source_feature)
        for (kind, _), operation in zip(self.program, self.operations):
            if kind == "pairwisecross":
                folded_batch, tokens, dim = latent.shape
                heads = self.projector.n_spaces
                latent = operation(
                    latent.reshape(folded_batch // heads, heads, tokens, dim)
                ).reshape(folded_batch, tokens, dim)
            else:
                latent = operation(latent)
        return self.projector.decode(mapping["score_decode"], latent)


class PANO(torch.nn.Module):
    """
    Point-space pre-norm blocks with latent interaction and a pointwise FFN.
    """

    def __init__(
        self,
        n_mode,
        n_dim,
        n_layer,
        coords_dim,
        condition_dim,
        sol_dim,
        act,
        *,
        time_proj=False,
        out_droprate=0.0,
        n_spaces=4,
        stage_specs,
        n_iter=1,
    ):
        super().__init__()
        if not stage_specs:
            raise ValueError("stage_specs must contain at least one named stage.")
        self.stage_specs = {
            name: tuple((tuple(op) for op in program))
            for name, program in stage_specs.items()
        }
        activation = ACTIVATION[act]
        self.initial_projector = PANOInitialProjector(
            n_dim, n_layer, condition_dim, activation
        )
        self.time_proj = time_proj
        if self.time_proj:
            self.time_projector = TimeProjector(n_dim, n_layer, activation)
        self.stages = torch.nn.ModuleList()
        self.point_norms = torch.nn.ModuleList()
        self.point_ffn_norms = torch.nn.ModuleList()
        self.point_ffns = torch.nn.ModuleList()
        self.point_drop = torch.nn.Dropout(0.0)
        for name, program in self.stage_specs.items():
            self.stages.append(
                PANOBlock(
                    name=name,
                    program=program,
                    n_mode=n_mode,
                    n_dim=n_dim,
                    n_head=8,
                    n_layer=3,
                    act=activation,
                    drop_rate=0.0,
                    n_spaces=n_spaces,
                    n_iter=n_iter,
                )
            )
            self.point_norms.append(torch.nn.LayerNorm(n_dim))
            self.point_ffn_norms.append(torch.nn.LayerNorm(n_dim))
            self.point_ffns.append(
                torch.nn.Sequential(
                    torch.nn.Linear(n_dim, 4 * n_dim),
                    activation,
                    torch.nn.Linear(4 * n_dim, n_dim),
                )
            )
        self.out_mlp = PANOMLP(
            n_dim, n_dim, sol_dim, 0, activation, drop_rate=out_droprate
        )
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize normalization, output/time MLPs, and attention projections.

        Routing and initial projectors retain their construction-time PyTorch
        initialization, matching the reference model's initialization order.
        """
        for module in self.modules():
            self._init_layer_norm(module)
        self._init_non_projector_mlps()
        self._init_attention_weights()

    def _init_layer_norm(self, module):
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.weight, 1.0)
            torch.nn.init.constant_(module.bias, 0)

    def _init_attention_linear(self, module):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)

    def _init_non_projector_mlps(self):
        for module in self.out_mlp.modules():
            if isinstance(module, torch.nn.Linear):
                self._init_attention_linear(module)
        if self.time_proj:
            for module in self.time_projector.modules():
                if isinstance(module, torch.nn.Linear):
                    self._init_attention_linear(module)

    def _init_attention_weights(self):
        initialized = set()
        for stage in self.stages:
            for stack in stage.initialization_stacks():
                for module in stack.modules():
                    if (
                        isinstance(module, torch.nn.Linear)
                        and id(module) not in initialized
                    ):
                        self._init_attention_linear(module)
                        initialized.add(id(module))
        for ffn in self.point_ffns:
            for module in ffn.modules():
                if isinstance(module, torch.nn.Linear):
                    self._init_attention_linear(module)

    def forward(self, x, y, t=None):
        """Build a feature-dependent routing map in each block."""
        point_feature = self.initial_projector(y)
        if t is not None and self.time_proj:
            point_feature = point_feature + self.time_projector(
                t, point_feature.shape[1]
            )
        for stage, norm, ffn_norm, ffn in zip(
            self.stages, self.point_norms, self.point_ffn_norms, self.point_ffns
        ):
            source_feature = norm(point_feature)
            decoded = stage(source_feature)
            decoded = self.point_drop(decoded)
            point_feature = point_feature + decoded
            update = ffn(ffn_norm(point_feature))
            point_feature = point_feature + update
        return self.out_mlp(point_feature)


def _build_pano(n_mode, n_dim, heads, depth):
    def builder(
        *,
        coords_dim,
        condition_dim,
        sol_dim,
        time_proj=False,
        out_droprate=0.0,
    ):
        program = (("attnlayer", depth), ("pairwisecross", 1), ("attnlayer", depth))
        return PANO(
            n_mode=n_mode,
            n_dim=n_dim,
            n_layer=3,
            coords_dim=coords_dim,
            condition_dim=condition_dim,
            sol_dim=sol_dim,
            act="GELU",
            time_proj=time_proj,
            out_droprate=out_droprate,
            n_spaces=heads,
            n_iter=1,
            stage_specs={"stage1": program, "stage2": program},
        )

    return builder


get_pano_p = _build_pano(512, 96, 4, 2)
get_pano_d = _build_pano(1024, 192, 2, 3)
