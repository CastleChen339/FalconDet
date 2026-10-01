import torch
import torch.nn.functional as F
import torch.nn as nn
import copy


def get_activation(act: str, inplace: bool = True):
    """
    Build an activation module by name.

    Args:
        act: Activation name.
        inplace: Whether to enable inplace mode when supported.

    Returns:
        Activation module.
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == "silu" or act == "swish":
        m = nn.SiLU()

    elif act == "relu":
        m = nn.ReLU()

    elif act == "leaky_relu":
        m = nn.LeakyReLU()

    elif act == "silu":
        m = nn.SiLU()

    elif act == "gelu":
        m = nn.GELU()

    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError("")

    if hasattr(m, "inplace"):
        m.inplace = inplace

    return m


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


# transformer
class AxisPermutedEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, q, k, v, src_mask=None) -> torch.Tensor:
        src = residual = v
        if self.normalize_before:
            src = self.norm1(src)

        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class AxisPermutedEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(AxisPermutedEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, glob_pos_embeds=None) -> torch.Tensor:
        """
        src: [B, L, C], L = D*h*w
        pos_embed: [C, h, w]
        glob_pos_embeds: [B, C, D, h, w]
        """
        B, L, C = src.shape
        assert glob_pos_embeds.dim() == 5, "glob_pos_embeds must be [B, C, D, h, w]"
        _, _, D, h, w = glob_pos_embeds.shape

        # flatten global embeddings to [B, L, C]
        glob_pos_flat = glob_pos_embeds.reshape(B, C, -1).permute(0, 2, 1)  # [B, L, C]

        # flatten relative position embedding [C, h, w] -> [1, L_per_depth, C]
        rel_pos_flat = pos_embed.reshape(C, -1).permute(1, 0).unsqueeze(0)  # [1, h*w, C]
        # broadcast rel_pos to all depth slices
        rel_pos_flat = rel_pos_flat.repeat(1, D, 1)  # [1, D*h*w, C]

        output = src
        for layer in self.layers:
            # Combine positional embeddings
            q = k = self.with_pos_embed(output, glob_pos_flat + rel_pos_flat)
            output = layer(q, k, output, src_mask=src_mask)

            # Alternate attention between depth and HW axes
            output = output.permute(1, 0, 2).contiguous()
            q = k = self.with_pos_embed(output, (glob_pos_flat + rel_pos_flat).permute(1, 0, 2).contiguous())
            output = layer(q, k, output, src_mask=src_mask)
            output = output.permute(1, 0, 2).contiguous()

        if self.norm is not None:
            output = self.norm(output)

        return output


class WindowProcessor(nn.Module):
    """
    Window-based attention processor for 3D feature maps.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dim_feedforward: Feedforward hidden size.
        num_layers: Number of encoder layers.
        dropout: Dropout probability.
        activation: Activation name.
    """
    def __init__(self, embed_dim=256, num_heads=8, dim_feedforward=1024, num_layers=1, dropout=0.0, activation="relu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Relative position encoder
        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

        encoder_layer = AxisPermutedEncoderLayer(
            self.embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.window_encoder = AxisPermutedEncoder(encoder_layer, self.num_layers)
        self.cross_window_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, backbone_memory, defe_feature_filtered, n, glob_pos_embed):
        """
        backbone_memory: [B, C, D, H, W]
        defe_feature_filtered: [B, 1, H, W]
        """

        B, C, D, H, W = backbone_memory.shape
        assert H % n == 0 and W % n == 0, "H and W must be divisible by n"

        rel_pos_embed = self._get_rel_embedding((H // n, W // n)).to(backbone_memory.device)
        reconstructed = backbone_memory.clone()
        windows, defe_mask = self._prepare_windows(backbone_memory, defe_feature_filtered, n)

        for b in range(B):
            valid_windows = torch.nonzero(defe_mask[b])
            if len(valid_windows) == 0:
                raise RuntimeError(f"No valid windows found for batch index {b}.")

            window_features, glob_pos_embeds = self._process_windows(
                windows[b], valid_windows, H, W, n, glob_pos_embed
            )

            encoded_features = self._encode_features(window_features, rel_pos_embed, glob_pos_embeds)

            self._reconstruct_features(
                reconstructed,
                b,
                encoded_features,
                valid_windows,
                H // n, W // n
            )

        return reconstructed, defe_mask

    def _prepare_windows(self, features, mask, n):
        """features: [B, C, D, H, W]"""
        B, C, D, H_feat, W_feat = features.shape
        H_mask, W_mask = mask.shape[-2:]

        # Compute pooling kernel size and stride
        kernel_h = H_mask // H_feat * (H_feat // n)
        kernel_w = W_mask // W_feat * (W_feat // n)
        stride_h = kernel_h
        stride_w = kernel_w

        # Partition feature map into windows: [B, n, n, C, D, h_feat, w_feat]
        h_feat, w_feat = H_feat // n, W_feat // n
        windows = features.view(B, C, D, n, h_feat, n, w_feat).permute(0, 3, 5, 1, 2, 4, 6)

        # Max-pool mask to determine valid windows
        mask_float = mask.float()  # pool requires floating dtype
        pooled_mask = F.max_pool2d(
            mask_float,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w)
        )
        defe_mask = (pooled_mask.squeeze(1) > 0)  # [B, n, n]

        return windows, defe_mask

    def _process_windows(self, windows, valid_indices, H, W, n, glob_pos_embed):
        batch_features = []
        batch_glob_pos_embed = []
        for i, j in valid_indices:
            win_feat = windows[i, j]  # [C, D, h, w]
            batch_features.append(win_feat.unsqueeze(0))
            batch_glob_pos_embed.append(
                self._get_abs_embedding(glob_pos_embed, i, j, win_feat.shape[2], win_feat.shape[3])
            )
        return torch.cat(batch_features, dim=0), torch.stack(batch_glob_pos_embed)

    def _encode_features(self, features, rel_pos_embed, glob_pos_embeds):
        """features: [N, C, D, h, w]"""
        N, C, D, h, w = features.shape
        # Flatten to tokens [N, L, C], L = D*h*w
        features = features.view(N, C, -1).permute(0, 2, 1)
        features = self.window_encoder(features, pos_embed=rel_pos_embed, glob_pos_embeds=glob_pos_embeds)
        # Cross-window global attention
        window_tokens = features.mean(dim=1, keepdim=True)
        window_tokens, _ = self.cross_window_attn(window_tokens, window_tokens, window_tokens)
        features = features + window_tokens.expand(-1, features.size(1), -1)
        return features.permute(0, 2, 1).view(N, C, D, h, w)

    def _reconstruct_features(self, reconstructed, batch_idx, feats, indices, win_h, win_w):
        for idx, (i, j) in enumerate(indices):
            h_start = i * win_h
            h_end = h_start + win_h
            w_start = j * win_w
            w_end = w_start + win_w
            reconstructed[batch_idx, :, :, h_start:h_end, w_start:w_end] += feats[idx]

    @staticmethod
    def _get_abs_embedding(global_emb, i, j, win_h, win_w):
        x0 = j * win_w
        y0 = i * win_h
        return global_emb[:, :, y0:y0 + win_h, x0:x0 + win_w]  # [C, D, win_h, win_w]

    def _get_rel_embedding(self, window_size):
        h, w = window_size
        coords = self._get_relative_coords(h, w)
        return self.rel_pos_encoder(coords.to(self.rel_pos_encoder[0].weight.device)).permute(2, 0, 1)

    @staticmethod
    def _get_relative_coords(h, w):
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        return torch.stack([grid_x / (w - 1), grid_y / (h - 1)], dim=-1)

