"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List
from torchvision.ops import roi_align

from .denoising import get_contrastive_denoising_training_group_3d
from .utils import get_activation, inverse_sigmoid, bias_init_with_prob, stabilize_box_cxcyczwhd
from ...core import register

__all__ = ['FalconDetTransformer']


def deformable_attention_core_func_v2( \
        value: torch.Tensor,
        value_spatial_shapes,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        num_points_list: List[int],
        method='default'):
    """
    Generalized deformable attention core supporting 2D and 3D sampling.

    Args:
        value: [bs, value_length, n_head, c]
        value_spatial_shapes: list/ Tensor of shape [n_levels, 2] (H,W) or [n_levels,3] (D,H,W)
        sampling_locations: [bs, query_length, n_head, n_levels * n_points, 2]  (2D)
                            or [..., 3] (3D)
        attention_weights: [bs, query_length, n_head, n_levels * n_points]
        num_points_list: list of int
    Returns:
        output: [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, last_dim = sampling_locations.shape

    # prepare value split per level
    if last_dim == 2:
        # 2D: value_spatial_shapes entries are (H,W)
        split_shape = [h * w for h, w in value_spatial_shapes]
    elif last_dim == 3:
        # 3D: value_spatial_shapes entries are (D,H,W)
        split_shape = [d * h * w for d, h, w in value_spatial_shapes]
    else:
        raise ValueError("sampling_locations last dim must be 2 or 3.")

    # value: [bs, value_length, n_head, c] -> permute to [bs, n_head, c, value_length]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    # prepare sampling grids per level
    if method == 'default':
        sampling_grids = 2 * sampling_locations - 1  # normalized -> grid_sample coordinate system (-1,1)
    elif method == 'discrete':
        sampling_grids = sampling_locations
    else:
        raise ValueError(f"unknown method {method}")

    # permute to [bs * n_head, Len_q, sum_points, lastdim]
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locs_per_level = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, shape in enumerate(value_spatial_shapes):
        if last_dim == 2:
            h, w = shape
            value_l = value_list[level].reshape(bs * n_head, c, h, w)  # [N, C, H, W]
            sampling_grid_l: torch.Tensor = sampling_locs_per_level[level]  # [N, Len_q, n_points, 2]

            if method == 'default':
                # grid_sample for 2D: input [N,C,H,W], grid [N, H_out, W_out, 2] -> out [N,C,H_out,W_out]
                # Here set H_out = Len_q, W_out = n_points
                grid = sampling_grid_l  # already [N, Len_q, n_points, 2]
                sampling_value_l = F.grid_sample(
                    value_l,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False)  # -> [N, C, Len_q, n_points]
            else:  # discrete
                # convert normalized coords to integer indices [x,y] in (W,H) order (consistent with earlier code)
                # sampling_grid_l in this branch expected to be normalized in [0,1) or absolute coords depending usage
                # We follow earlier convention: sampling_grid_l * [w,h] + 0.5
                wh = torch.tensor([[w, h]], device=value.device, dtype=torch.int64)
                sampling_coord = (sampling_grid_l * wh.float() + 0.5).to(torch.int64)  # [N, Len_q, n_points, 2]
                sampling_coord = sampling_coord.clamp_min(0)
                sampling_coord[..., 0] = sampling_coord[..., 0].clamp_max(w - 1)
                sampling_coord[..., 1] = sampling_coord[..., 1].clamp_max(h - 1)
                sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)
                s_idx = (torch.arange(sampling_coord.shape[0], device=value.device)
                         .unsqueeze(-1).repeat(1, sampling_coord.shape[1]))
                sampling_value_l = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]  # [N, seq, C]
                sampling_value_l = (sampling_value_l.permute(0, 2, 1)
                                    .reshape(bs * n_head, c, Len_q, num_points_list[level]))

            sampling_value_list.append(sampling_value_l)


        else:  # lastdim == 3 -> 3D
            d, h, w = shape  # value_spatial_shapes expected (D,H,W)
            # value_l as [N, C, D, H, W]
            value_l = value_list[level].reshape(bs * n_head, c, d, h, w)
            sampling_grid_l: torch.Tensor = sampling_locs_per_level[level]  # [N, Len_q, n_points, 3]

        if method == 'default':
            # grid for 3D: shape [N, D_out, H_out, W_out, 3]
            # we'll set D_out = Len_q, H_out = n_points, W_out = 1
            grid = sampling_grid_l.unsqueeze(-2)  # -> [N, Len_q, n_points, 1, 3]
            sampling_value_l_5d = F.grid_sample(
                value_l,
                grid,
                mode='bilinear',  # trilinear for 3D input
                padding_mode='zeros',
                align_corners=False)  # -> [N, C, Len_q, n_points, 1]
            sampling_value_l = sampling_value_l_5d.squeeze(-1)  # -> [N, C, Len_q, n_points]

        else:  # discrete for 3D
            # sampling_grid_l is in normalized coords; convert to integer voxel indices
            whd = torch.tensor([[w, h, d]], device=value.device, dtype=torch.int64)  # careful order
            # We need to ensure coordinate ordering: grid values assumed (x,y,z) mapping to (W,H,D)
            sampling_coord = (sampling_grid_l * whd.float() + 0.5).to(torch.int64)  # [N, Len_q, n_points, 3]
            # clamp (x->W-1, y->H-1, z->D-1)
            sampling_coord[..., 0] = sampling_coord[..., 0].clamp_min(0).clamp_max(w - 1)
            sampling_coord[..., 1] = sampling_coord[..., 1].clamp_min(0).clamp_max(h - 1)
            sampling_coord[..., 2] = sampling_coord[..., 2].clamp_min(0).clamp_max(d - 1)

            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 3)
            s_idx = torch.arange(sampling_coord.shape[0], device=value.device).unsqueeze(-1).repeat(1,
                                                                                                    sampling_coord.shape[
                                                                                                        1])
            # index value_l at [N, :, z, y, x] -> note indexing order [z,y,x]
            sampling_value_l = value_l[
                s_idx, :, sampling_coord[..., 2], sampling_coord[..., 1], sampling_coord[..., 0]
            ]  # -> [N, seq, C]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q,
                                                                         num_points_list[level])

        sampling_value_list.append(sampling_value_l)

    # combine levels
    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights  # [N, C, Len_q, sum_points]
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)  # -> [bs, Len_q, C_out]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            method="default",
            offset_scale=0.5,
            spatial_ndim=3,  # 2 for 2D (x,y) sampling, 3 for 3D (x,y,z) sampling
    ):
        """Multi-Scale Deformable Attention that supports 2D and 3D sampling.

        Args:
            spatial_ndim: 2 or 3. When 3, sampling offsets and reference points are 3-dimensional.
        """
        super(MSDeformableAttention, self).__init__()
        assert spatial_ndim in (2, 3), "spatial_ndim must be 2 or 3"
        self.spatial_ndim = spatial_ndim

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, "num_points list length must equal num_levels"
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # sampling_offsets outputs 2 or 3 coordinates per sampling point depending on spatial_ndim
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * self.spatial_ndim)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=self.method
        )

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)

        # Initialize bias with directional vectors (ring for 2D, sphere for 3D).
        sum_points = sum(self.num_points_list)
        if self.spatial_ndim == 2:
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # [n_heads, 2]
            grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
            grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum_points, 1])

        else:  # 3D
            # Simple spherical parameterization: head -> azimuth (theta), point -> elevation (phi).
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / max(1, self.num_heads))
            # Distribute phi linearly in (0, pi) to avoid poles.
            phis = torch.arange(1, sum_points + 1, dtype=torch.float32) * (math.pi / (sum_points + 1))
            # Expand to grid
            thetas = thetas.unsqueeze(1).repeat(1, sum_points)  # [n_heads, sum_points]
            phis = phis.unsqueeze(0).repeat(self.num_heads, 1)  # [n_heads, sum_points]
            x = (phis.sin() * thetas.cos())
            y = (phis.sin() * thetas.sin())
            z = phis.cos()
            grid_init = torch.stack([x, y, z], dim=-1)  # [n_heads, sum_points, 3]
            # Normalize and apply radial scaling
            grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values

        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling  # radial scaling by point index
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(
            self,
            query: torch.Tensor,
            reference_points: torch.Tensor,
            value: torch.Tensor,
            value_spatial_shapes: List[int],
            value_mask: torch.Tensor = None
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): if spatial_ndim==2 -> [bs, query_length, n_levels, 2] (x,y)
                if spatial_ndim==3 -> [bs, query_length, n_levels, 3] (x,y,z)
                Coordinates are normalized to [0,1] (top-left/near (0,0,0) - bottom-right/far (1,1,1)).
                Optionally reference_points may include sizes: 2D -> 4 ([x,y,w,h]), 3D -> 6 ([x,y,z,dx,dy,dz]).
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [(H0,W0) ...] for 2D or [(D0,H0,W0), ...] for 3D

        Returns:
        output (Tensor): [bs, Length_{query}, C]
        """

        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list),
                                                    self.spatial_ndim)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads,
                                                                         sum(self.num_points_list))

        if reference_points.shape[-1] == self.spatial_ndim:
            # normalization: convert value_spatial_shapes -> normalizer in order (W,H) or (W,H,D)
            offset_normalizer = torch.tensor(value_spatial_shapes, dtype=reference_points.dtype,
                                             device=reference_points.device)
            # flip last dimension order so that reference_points (x,y,(z)) divides by (W,H,(D))
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, self.spatial_ndim)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1,
                                                          self.spatial_ndim) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 2 * self.spatial_ndim:
            # box form: (cx,cy,(cz), w,h,(d))
            # scale sampling offsets by box size
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            # reference_points[..., :self.spatial_ndim] = center, [..., self.spatial_ndim:] = size
            offset = sampling_offsets * num_points_scale * reference_points[
                :, :, None, :, self.spatial_ndim:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :self.spatial_ndim] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be {} or {}, but get {} instead.".
                format(self.spatial_ndim, 2 * self.spatial_ndim, reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )
        output = self.output_proj(output)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn( \
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask,
                           query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class TubeRoIAlign(nn.Module):
    """
    Fully vectorized Tube RoI Align with temporal interpolation.

    Input:
        feat:  [B, C, T, H, W]
        boxes: [B, N, 6] in (cx, cy, cz, w, h, d), normalized to [0, 1]

    Output:
        tube_feat: [B, N, C, T', H', W']
    """

    def __init__(
        self,
        out_size=(8, 8),
        temporal_bins=5,
    ):
        super().__init__()
        self.out_h, self.out_w = out_size
        self.temporal_bins = temporal_bins

    @staticmethod
    def _pixel_to_norm(x: torch.Tensor, size: int):
        """
        Convert pixel index (continuous) -> normalized [-1, 1]
        Exactly matches grid_sample with align_corners=False.
        """
        return (2.0 * (x + 0.5) / size) - 1.0

    def forward(self, feat: torch.Tensor, boxes: torch.Tensor):
        device = feat.device
        dtype = feat.dtype

        B, C, T, H, W = feat.shape
        _, N, _ = boxes.shape
        BN = B * N

        # --------------------------------------------------------
        # Step 1: flatten boxes & build spatial RoIs
        # --------------------------------------------------------
        boxes_flat = boxes.reshape(BN, 6)
        cx, cy, cz, w, h, d = boxes_flat.unbind(dim=1)
        # During JIT-based FLOP tracing, shape values can be represented as
        # CPU tensors.  Materialize the lower bounds on the feature device so
        # clamp_min does not mix CPU shape constants with CUDA box tensors.
        min_w = torch.as_tensor(2.0 / max(W, 1), device=device, dtype=dtype)
        min_h = torch.as_tensor(2.0 / max(H, 1), device=device, dtype=dtype)
        min_d = torch.as_tensor(1.0 / max(T, 1), device=device, dtype=dtype)
        w = w.clamp_min(min_w)
        h = h.clamp_min(min_h)
        d = d.clamp_min(min_d)

        x1 = (cx - 0.5 * w) * (W - 1)
        y1 = (cy - 0.5 * h) * (H - 1)
        x2 = (cx + 0.5 * w) * (W - 1)
        y2 = (cy + 0.5 * h) * (H - 1)

        batch_ids = (
            torch.arange(B, device=device)
            .view(B, 1)
            .expand(B, N)
            .reshape(-1)
        )

        rois = torch.stack([batch_ids, x1, y1, x2, y2], dim=1)  # [BN, 5]

        # --------------------------------------------------------
        # Step 2: collapse time into batch, vectorized roi_align
        # --------------------------------------------------------
        # feat: [B, C, T, H, W] -> [B*T, C, H, W]
        feat_bt = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # repeat rois for each time step
        rois_t = rois[:, None, :].expand(BN, T, 5).reshape(BN * T, 5).clone()

        # adjust batch index for time offset
        time_offsets = (
            torch.arange(T, device=device)
            .view(1, T)
            .expand(BN, T)
            .reshape(-1)
        )
        rois_t[:, 0] = rois_t[:, 0] * T + time_offsets

        # single roi_align call
        pooled = roi_align(
            feat_bt,
            rois_t,
            output_size=(self.out_h, self.out_w),
            spatial_scale=1.0,
            aligned=True,
        )  # [BN*T, C, H', W']

        # reshape -> [BN, C, T, H', W']
        feat_tube = (
            pooled
            .view(BN, T, C, self.out_h, self.out_w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        # --------------------------------------------------------
        # Step 3: temporal interpolation (3D grid_sample)
        # --------------------------------------------------------
        # time range in pixel coordinates [0, T-1]
        t_start = (cz - 0.5 * d) * (T - 1)
        t_end   = (cz + 0.5 * d) * (T - 1)
        # optional safety clamp
        t_start = t_start.clamp(0, T - 1)
        t_end   = t_end.clamp(0, T - 1)

        t_steps = torch.linspace(
            0, 1, self.temporal_bins,
            device=device, dtype=dtype
        )

        # continuous sampling positions
        t_samples = (
            t_start[:, None]
            + t_steps[None, :] * (t_end - t_start)[:, None]
        )  # [BN, T']

        # normalize to [-1, 1], exactly matching grid_sample
        t_norm = self._pixel_to_norm(t_samples, T)  # [BN, T']

        # spatial identity grid
        h_grid = torch.linspace(-1, 1, self.out_h, device=device, dtype=dtype)
        w_grid = torch.linspace(-1, 1, self.out_w, device=device, dtype=dtype)
        h_grid, w_grid = torch.meshgrid(h_grid, w_grid, indexing="ij")

        h_grid = h_grid[None, None].expand(BN, self.temporal_bins, -1, -1)
        w_grid = w_grid[None, None].expand(BN, self.temporal_bins, -1, -1)
        t_grid = t_norm[:, :, None, None].expand(-1, -1, self.out_h, self.out_w)

        # grid[..., 0]=x, 1=y, 2=z(time)
        grid = torch.stack([w_grid, h_grid, t_grid], dim=-1)
        # [BN, T', H', W', 3]

        tube_interp = F.grid_sample(
            feat_tube,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [BN, C, T', H', W']

        # --------------------------------------------------------
        # Step 4: reshape back to [B, N, ...]
        # --------------------------------------------------------
        tube_interp = tube_interp.view(
            B, N, C, self.temporal_bins, self.out_h, self.out_w
        )

        return tube_interp


class TubePointDecoder(nn.Module):
    """
    Tube-conditioned point decoder.
    Each query corresponds to one temporal step (frame).
    """

    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_layers=3,
            dim_feedforward=1024,
            dropout=0.0,
            num_points=5,  # temporal_bins or seq_len
    ):
        super().__init__()
        self.num_points = num_points
        self.max_time_offset = 0.5 / max(num_points, 1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.memory_pos_embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "query_time",
            (torch.arange(num_points, dtype=torch.float32) + 0.5) / max(num_points, 1),
        )

        # One query corresponds to one temporal bin.
        self.point_query_embed = nn.Embedding(num_points, d_model)

        # Regress (x, y, dt) and score.
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4)
        )

    def forward(self, tube_feat):
        """
        tube_feat: [BN, C, T', H', W']
        return:
            points: [BN, T', 3]  -> (x, y, t)
            scores: [BN, T']     -> confidence
        """
        BN, C, T, H, W = tube_feat.shape

        # flatten tube feature -> memory
        memory = tube_feat.flatten(3).permute(0, 2, 3, 1)
        # [BN, T, H*W, C] -> merge T,H,W
        memory = memory.reshape(BN, T * H * W, C)

        # Explicit spatiotemporal coordinates help the point decoder keep
        # track of where each memory token came from after flattening.
        t_coord = (torch.arange(T, device=tube_feat.device, dtype=tube_feat.dtype) + 0.5) / max(T, 1)
        h_coord = (torch.arange(H, device=tube_feat.device, dtype=tube_feat.dtype) + 0.5) / max(H, 1)
        w_coord = (torch.arange(W, device=tube_feat.device, dtype=tube_feat.dtype) + 0.5) / max(W, 1)
        tt, yy, xx = torch.meshgrid(t_coord, h_coord, w_coord, indexing="ij")
        memory_coords = torch.stack([xx, yy, tt], dim=-1).reshape(1, T * H * W, 3).expand(BN, -1, -1)
        memory = memory + self.memory_pos_embed(memory_coords)

        # point queries
        time_query = self.query_time.to(device=tube_feat.device, dtype=tube_feat.dtype).view(1, self.num_points, 1)
        tube_context = tube_feat.mean(dim=(2, 3, 4))
        query = self.point_query_embed.weight.unsqueeze(0).expand(BN, -1, -1) + self.time_embed(time_query)
        query = query + self.context_proj(tube_context).unsqueeze(1)
        # [BN, T, C]

        # Transformer decoder
        out = self.output_norm(self.decoder(
            tgt=query,
            memory=memory,
        ))  # [BN, T, C]

        raw = self.point_head(out)  # [BN, T, 4]

        points = raw[..., 0:3].sigmoid()  # (x, y, t) in [0,1]
        xy = raw[..., 0:2].sigmoid()
        t = (time_query.squeeze(-1) + raw[..., 2].tanh() * self.max_time_offset).clamp(0.0, 1.0)
        points = torch.cat([xy, t.unsqueeze(-1)], dim=-1)
        score_logits = raw[..., 3]
        score = score_logits.sigmoid()  # point confidence

        return points, score, score_logits


@register()
class FalconDetTransformer(nn.Module):
    """
    Transformer decoder for tube and point predictions.
    """
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 depth_strides=[1, 1, 1],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 point_train_box_jitter_xy=0.0,
                 point_train_box_jitter_t=0.0,
                 point_train_box_jitter_wh=0.0,
                 point_train_box_jitter_d=0.0,
                 point_train_use_gt=True,
                 point_train_use_jitter=True):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.depth_strides = depth_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.point_train_box_jitter_xy = point_train_box_jitter_xy
        self.point_train_box_jitter_t = point_train_box_jitter_t
        self.point_train_box_jitter_wh = point_train_box_jitter_wh
        self.point_train_box_jitter_d = point_train_box_jitter_d
        self.point_train_use_gt = point_train_use_gt
        self.point_train_use_jitter = point_train_use_jitter

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
                                                activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(6, 2 * hidden_dim, hidden_dim, 2)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim, )),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 6, 3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 6, 3) for _ in range(num_layers)
        ])

        # ----------------------------
        # Tube -> Point head
        # ----------------------------
        self.tube_roi_align = TubeRoIAlign(
            out_size=(8, 8),
            temporal_bins=self.eval_spatial_size[0] if self.eval_spatial_size else 5,
        )

        self.tube_point_decoder = TubePointDecoder(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=3,
            num_points=self.eval_spatial_size[0] if self.eval_spatial_size else 5,
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm3d(self.hidden_dim, ))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm3d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _apply_point_train_box_jitter(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes

        boxes = boxes.clone()
        centers = boxes[:, :3]
        sizes = boxes[:, 3:]

        if self.point_train_box_jitter_xy > 0:
            centers[:, :2] = centers[:, :2] + torch.randn_like(centers[:, :2]) * sizes[:, :2] * self.point_train_box_jitter_xy
        if self.point_train_box_jitter_t > 0:
            centers[:, 2] = centers[:, 2] + torch.randn_like(centers[:, 2]) * sizes[:, 2] * self.point_train_box_jitter_t

        if self.point_train_box_jitter_wh > 0:
            scale_xy = 1.0 + torch.randn_like(sizes[:, :2]) * self.point_train_box_jitter_wh
            sizes[:, :2] = sizes[:, :2] * scale_xy.clamp(0.6, 1.5)
        if self.point_train_box_jitter_d > 0:
            scale_t = 1.0 + torch.randn_like(sizes[:, 2]) * self.point_train_box_jitter_d
            sizes[:, 2] = sizes[:, 2] * scale_t.clamp(0.6, 1.5)

        jittered = torch.cat([centers, sizes], dim=-1)
        jittered = stabilize_box_cxcyczwhd(jittered)
        half_size = jittered[:, 3:] * 0.5
        jittered[:, :3] = torch.minimum(torch.maximum(jittered[:, :3], half_size), 1.0 - half_size)
        return stabilize_box_cxcyczwhd(jittered)

    def _decode_point_boxes(
            self,
            feature: torch.Tensor,
            boxes: torch.Tensor,
            valid_mask: torch.Tensor = None):
        """Decode tube-relative points for a padded batch of tube boxes."""
        batch_size, num_boxes = boxes.shape[:2]
        if num_boxes == 0:
            empty = [None for _ in range(batch_size)]
            return empty, empty.copy(), empty.copy(), empty.copy()

        tube_feats = self.tube_roi_align(feature, boxes)
        _, _, channels, temporal_bins, height, width = tube_feats.shape
        flat_feats = tube_feats.reshape(
            batch_size * num_boxes,
            channels,
            temporal_bins,
            height,
            width,
        )
        points, scores, logits = self.tube_point_decoder(flat_feats)
        points = points.reshape(batch_size, num_boxes, temporal_bins, 3)
        scores = scores.reshape(batch_size, num_boxes, temporal_bins)
        logits = logits.reshape(batch_size, num_boxes, temporal_bins)

        point_list = []
        score_list = []
        logit_list = []
        box_list = []
        for batch_index in range(batch_size):
            mask = (
                torch.ones(num_boxes, dtype=torch.bool, device=boxes.device)
                if valid_mask is None
                else valid_mask[batch_index]
            )
            if mask.any():
                point_list.append(points[batch_index][mask])
                score_list.append(scores[batch_index][mask])
                logit_list.append(logits[batch_index][mask])
                box_list.append(boxes[batch_index][mask])
            else:
                point_list.append(None)
                score_list.append(None)
                logit_list.append(None)
                box_list.append(None)
        return point_list, score_list, logit_list, box_list

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, d, h, w = feat.shape
            # [b, c, d, h, w] -> [b, d*h*w, c]
            feat_flatten.append(feat.flatten(2, 4).permute(0, 2, 1))
            # [num_levels, 3]
            spatial_shapes.append([d, h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_d, eval_h, eval_w = self.eval_spatial_size
            for i in range(len(self.feat_strides)):
                s = self.feat_strides[i]
                d_s = self.depth_strides[i]
                spatial_shapes.append([int(eval_d / d_s), int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (d, h, w) in enumerate(spatial_shapes):
            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.arange(d, device=device),
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            grid_xyz = (grid_xyz + 0.5) / torch.tensor([w, h, d], device=device, dtype=dtype)
            whd = torch.ones_like(grid_xyz) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xyz, whd], dim=-1).reshape(-1, d * h * w, 6)

            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)
        enc_outputs_coord_unact: torch.Tensor = self.enc_bbox_head(output_memory) + anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor,
                     topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1,
                                                  index=topk_ind.unsqueeze(-1).repeat(1, 1,
                                                                                      outputs_coords_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1,
                                            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))

        topk_memory = memory.gather(dim=1,
                                    index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords

    def forward(self, feats, targets=None):
        """
        Forward pass for tube and point prediction.

        Args:
            feats: List of feature maps from the encoder.
            targets: Optional targets for training.

        Returns:
            Dict of prediction tensors and optional auxiliary outputs.
        """
        # ------------------------------------------------
        # Level-1: Tube detection
        # ------------------------------------------------
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group_3d(targets,
                                                            self.num_classes,
                                                            self.num_queries,
                                                            self.denoising_class_embed,
                                                            num_denoising=self.num_denoising,
                                                            label_noise_ratio=self.label_noise_ratio,
                                                            box_noise_scale=self.box_noise_scale
                                                            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        tube_boxes = out_bboxes[-1]
        tube_logits = out_logits[-1]

        # ------------------------------------------------
        # Level-2: Tube -> Point
        # ------------------------------------------------
        point_pred_boxes = tube_boxes.detach()
        pred_points, pred_scores, pred_score_logits, pred_point_boxes = (
            self._decode_point_boxes(feats[0], point_pred_boxes)
        )

        point_gt_outputs = None
        point_jitter_outputs = None
        if self.training:
            batch_size = len(targets)
            device = feats[-1].device
            max_num = max((target['bboxes'].shape[0] for target in targets), default=0)
            gt_boxes = torch.zeros(batch_size, max_num, 6, device=device)
            gt_mask = torch.zeros(
                batch_size, max_num, device=device, dtype=torch.bool
            )
            for batch_index, target in enumerate(targets):
                count = target['bboxes'].shape[0]
                if count:
                    gt_boxes[batch_index, :count] = target['bboxes']
                    gt_mask[batch_index, :count] = True

            if self.point_train_use_gt and max_num > 0:
                points, scores, logits, boxes = self._decode_point_boxes(
                    feats[0], gt_boxes, gt_mask
                )
                point_gt_outputs = {
                    'pred_points': points,
                    'pred_point_scores': scores,
                    'pred_point_logits': logits,
                    'boxes': boxes,
                }

            use_jitter = self.point_train_use_jitter and any(v > 0 for v in (
                self.point_train_box_jitter_xy,
                self.point_train_box_jitter_t,
                self.point_train_box_jitter_wh,
                self.point_train_box_jitter_d,
            ))
            if use_jitter and max_num > 0:
                jitter_boxes = gt_boxes.clone()
                for batch_index in range(batch_size):
                    if gt_mask[batch_index].any():
                        jitter_boxes[batch_index, gt_mask[batch_index]] = (
                            self._apply_point_train_box_jitter(
                                jitter_boxes[batch_index, gt_mask[batch_index]]
                            )
                        )
                points, scores, logits, boxes = self._decode_point_boxes(
                    feats[0], jitter_boxes, gt_mask
                )
                point_jitter_outputs = {
                    'pred_points': points,
                    'pred_point_scores': scores,
                    'pred_point_logits': logits,
                    'boxes': boxes,
                }


        out = {
            'pred_logits': tube_logits, # Tensor, shape [B, Q, num_classes]
            'pred_boxes': tube_boxes, # Tensor, shape [B, Q, 6]
            'pred_points': pred_points, # List[Optional[Tensor]]
            'pred_point_scores': pred_scores, # List[Optional[Tensor]]
            'pred_point_logits': pred_score_logits, # List[Optional[Tensor]]
            'point_pred_boxes': pred_point_boxes,
        }
        if point_gt_outputs is not None:
            out['point_gt_outputs'] = point_gt_outputs
        if point_jitter_outputs is not None:
            out['point_jitter_outputs'] = point_jitter_outputs

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
