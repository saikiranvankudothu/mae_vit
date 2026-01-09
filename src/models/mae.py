"""
Masked Autoencoder (MAE) Implementation for Chest X-ray Denoising
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple, Optional


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0., attn_dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, ids_keep, ids_restore):
        x = self.patch_embed(x)
        B, N, D = x.shape
        x = x + self.pos_embed[:, 1:, :]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class MAEDecoder(nn.Module):
    def __init__(self, num_patches=196, patch_size=16, in_chans=1, embed_dim=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, L, D = x.shape
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - L, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x


class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_pix_loss=True, mask_ratio=0.75):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.encoder = MAEEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio)
        num_patches = (img_size // patch_size) ** 2
        self.decoder = MAEDecoder(num_patches, patch_size, in_chans, embed_dim,
                                  decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio)

    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * self.in_chans)
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_chans)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], self.in_chans, h * p, w * p)
        return imgs

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B, C, H, W = imgs.shape
        num_patches = (H // self.patch_size) ** 2
        mask, ids_restore = self.random_masking(torch.zeros(B, num_patches, 1, device=imgs.device), mask_ratio)
        len_keep = int(num_patches * (1 - mask_ratio))
        ids_shuffle = torch.argsort(mask, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        latent = self.encoder(imgs, ids_keep, ids_restore)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        pred_imgs = self.unpatchify(pred)
        return loss, pred_imgs, mask

    @torch.no_grad()
    def denoise(self, imgs, mask_ratio=0.0):
        self.eval()
        B, C, H, W = imgs.shape
        num_patches = (H // self.patch_size) ** 2
        if mask_ratio > 0:
            mask, ids_restore = self.random_masking(torch.zeros(B, num_patches, 1, device=imgs.device), mask_ratio)
            len_keep = int(num_patches * (1 - mask_ratio))
            ids_shuffle = torch.argsort(mask, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
        else:
            ids_restore = torch.arange(num_patches, device=imgs.device).unsqueeze(0).expand(B, -1)
            ids_keep = ids_restore
        latent = self.encoder(imgs, ids_keep, ids_restore)
        pred = self.decoder(latent, ids_restore)
        denoised_imgs = self.unpatchify(pred)
        return denoised_imgs


def build_mae_model(config):
    model_cfg = config['model']
    model = MaskedAutoencoder(
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size'],
        in_chans=model_cfg['in_channels'],
        embed_dim=model_cfg['encoder']['embed_dim'],
        depth=model_cfg['encoder']['depth'],
        num_heads=model_cfg['encoder']['num_heads'],
        decoder_embed_dim=model_cfg['decoder']['decoder_embed_dim'],
        decoder_depth=model_cfg['decoder']['decoder_depth'],
        decoder_num_heads=model_cfg['decoder']['decoder_num_heads'],
        mlp_ratio=model_cfg['encoder']['mlp_ratio'],
        norm_pix_loss=model_cfg['norm_pix_loss'],
        mask_ratio=model_cfg['mask_ratio']
    )
    return model
