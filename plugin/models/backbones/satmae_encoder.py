"""
Self-contained SatMAE ViT-Large encoder for satellite feature extraction.
No dependency on SatMAE repo or specific timm version.

Loads pretrained weights from fMoW non-temporal checkpoint.
Outputs patch tokens that can be used as K/V in cross-attention.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import BACKBONES


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sine-cosine positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention (compatible with SatMAE checkpoint keys)."""
    def __init__(self, dim, num_heads=16, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP block (compatible with SatMAE checkpoint keys: mlp.fc1, mlp.fc2)."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block (compatible with SatMAE checkpoint keys)."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@BACKBONES.register_module()
class SatMAEEncoder(nn.Module):
    """SatMAE ViT-Large encoder for satellite feature extraction.

    Loads pretrained weights from fMoW non-temporal MAE checkpoint.
    Outputs patch tokens as (B, num_patches, embed_dim).

    Args:
        img_size (int): Input image size (resized to square). Default: 224.
        patch_size (int): Patch size. Default: 16.
        in_chans (int): Input channels. Default: 3.
        embed_dim (int): Embedding dimension. Default: 1024.
        depth (int): Number of transformer blocks. Default: 24.
        num_heads (int): Number of attention heads. Default: 16.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.
        out_channels (int): Output projection channels. Default: 256.
        bev_size (tuple): Target BEV spatial size (H, W). Default: (50, 100).
        pretrained (str): Path to pretrained checkpoint. Default: None.
        frozen (bool): Whether to freeze encoder weights. Default: True.
    """

    # fMoW-RGB normalization (NOT ImageNet!)
    FMOW_MEAN = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    FMOW_STD = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 out_channels=256,
                 bev_size=(50, 100),
                 pretrained=None,
                 frozen=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.bev_size = bev_size
        self.frozen = frozen

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = int(num_patches ** 0.5)

        # CLS token + fixed sincos positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Projection: embed_dim -> out_channels
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.LayerNorm(out_channels),
        )

        # Initialize positional encoding
        self._init_pos_embed()

        # Load pretrained weights
        if pretrained is not None:
            self._load_pretrained(pretrained)

        if frozen:
            self._freeze_encoder()

        # Register normalization buffers
        self.register_buffer('pixel_mean',
            torch.tensor(self.FMOW_MEAN).view(1, 3, 1, 1))
        self.register_buffer('pixel_std',
            torch.tensor(self.FMOW_STD).view(1, 3, 1, 1))

    def _init_pos_embed(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.embed_dim, self.grid_size, cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _load_pretrained(self, checkpoint_path):
        print(f'Loading SatMAE pretrained weights from {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt

        # Filter: only encoder keys (no decoder_*, no mask_token)
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('decoder') or k == 'mask_token':
                continue
            encoder_state[k] = v

        # Handle pos_embed size mismatch (different input resolution)
        if 'pos_embed' in encoder_state:
            ckpt_pos = encoder_state['pos_embed']
            if ckpt_pos.shape != self.pos_embed.shape:
                print(f'Interpolating pos_embed: {ckpt_pos.shape} -> {self.pos_embed.shape}')
                # Separate cls token
                ckpt_cls = ckpt_pos[:, :1, :]
                ckpt_spatial = ckpt_pos[:, 1:, :]
                orig_size = int(ckpt_spatial.shape[1] ** 0.5)
                new_size = self.grid_size
                ckpt_spatial = ckpt_spatial.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                ckpt_spatial = F.interpolate(ckpt_spatial, size=(new_size, new_size),
                                             mode='bicubic', align_corners=False)
                ckpt_spatial = ckpt_spatial.permute(0, 2, 3, 1).flatten(1, 2)
                encoder_state['pos_embed'] = torch.cat([ckpt_cls, ckpt_spatial], dim=1)

        msg = self.load_state_dict(encoder_state, strict=False)
        print(f'  Missing keys: {msg.missing_keys}')
        print(f'  Unexpected keys: {msg.unexpected_keys}')

    def _freeze_encoder(self):
        """Freeze all encoder parameters, keep projection trainable."""
        for name, param in self.named_parameters():
            if name.startswith('proj.'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f'SatMAE encoder frozen. Trainable params: proj layer only.')

    def _normalize_input(self, sat_img):
        """Normalize satellite image with fMoW statistics.

        Args:
            sat_img: (B, 3, H, W) in [0, 1] range (after ToTensor).
        Returns:
            Normalized tensor.
        """
        return (sat_img - self.pixel_mean) / self.pixel_std

    def forward(self, sat_img):
        """
        Args:
            sat_img (Tensor): (B, 3, H, W) satellite image.
                Expected to be in [0, 1] range (standard ToTensor output).
        Returns:
            patch_tokens (Tensor): (B, num_patches, out_channels)
                Spatial patch token features projected to out_channels.
            grid_size (int): Spatial grid dimension (tokens are grid_size x grid_size).
        """
        B = sat_img.shape[0]

        # Resize to model input size
        if sat_img.shape[2:] != (self.img_size, self.img_size):
            sat_img = F.interpolate(sat_img, size=(self.img_size, self.img_size),
                                    mode='bilinear', align_corners=False)

        # Normalize with fMoW statistics
        sat_img = self._normalize_input(sat_img)

        # Patch embedding
        x = self.patch_embed(sat_img)  # (B, num_patches, embed_dim)

        # Add positional encoding (skip cls token slot)
        x = x + self.pos_embed[:, 1:, :]

        # Prepend cls token
        cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Extract patch tokens (drop cls token)
        patch_tokens = x[:, 1:, :]  # (B, num_patches, embed_dim)

        # Project to out_channels
        patch_tokens = self.proj(patch_tokens)  # (B, num_patches, out_channels)

        # H-flip: match BEVFormer's inverted y-axis convention
        C = patch_tokens.shape[-1]
        patch_tokens = patch_tokens.reshape(B, self.grid_size, self.grid_size, C)
        patch_tokens = torch.flip(patch_tokens, [1])
        patch_tokens = patch_tokens.reshape(B, self.grid_size * self.grid_size, C)

        return patch_tokens, self.grid_size
