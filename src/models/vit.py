"""
Vision Transformer (ViT) Implementation for Tuberculosis Classification

This module implements the Vision Transformer architecture based on:
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2020)

Adapted for binary classification of chest X-rays (TB vs Normal).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
import math


class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them.

    Args:
        img_size (int): Input image size
        patch_size (int): Size of each patch
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, D) patch embeddings
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class Attention(nn.Module):
    """
    Multi-head Self-Attention with optional attention map extraction.

    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        attn_dropout (float): Attention dropout rate
        proj_dropout (float): Projection dropout rate
    """
    def __init__(self, dim: int, num_heads: int = 8,
                 attn_dropout: float = 0., proj_dropout: float = 0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        # Store attention weights for visualization
        self.attn_weights = None

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            return_attention: Whether to store attention weights
        Returns:
            (B, N, D) or ((B, N, D), (B, num_heads, N, N))
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if return_attention:
            self.attn_weights = attn.detach()

        attn = self.attn_drop(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class MLP(nn.Module):
    """
    MLP block with GELU activation.
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, dropout: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with pre-normalization.

    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP hidden dim ratio
        dropout (float): Dropout rate
        attn_dropout (float): Attention dropout rate
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 dropout: float = 0., attn_dropout: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for TB Classification.

    Architecture:
    1. Patch Embedding
    2. Positional Embedding
    3. Transformer Encoder Blocks
    4. Classification Head

    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        in_chans (int): Input channels (1 for grayscale)
        num_classes (int): Number of output classes (2 for binary)
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP hidden dimension ratio
        dropout (float): Dropout rate
        attn_dropout (float): Attention dropout rate
        drop_path_rate (float): Stochastic depth rate
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 1,
                 num_classes: int = 2, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4., dropout: float = 0.,
                 attn_dropout: float = 0., drop_path_rate: float = 0.):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # Initialize classification head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

        # Initialize other layers
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        """Initialize weights for linear and layer norm layers."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor, return_attention: bool = False):
        """
        Extract features from input.

        Args:
            x: (B, C, H, W) input images
            return_attention: Whether to return attention maps

        Returns:
            (B, D) cls token features or (features, attention_maps)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        attention_maps = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x = block(x)

        x = self.norm(x)

        # Return cls token
        if return_attention:
            return x[:, 0], attention_maps
        return x[:, 0]

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass for classification.

        Args:
            x: (B, C, H, W) input images
            return_attention: Whether to return attention maps

        Returns:
            logits: (B, num_classes) class logits
            or (logits, attention_maps) if return_attention=True
        """
        if return_attention:
            features, attention_maps = self.forward_features(x, return_attention=True)
            logits = self.head(features)
            return logits, attention_maps
        else:
            features = self.forward_features(x)
            logits = self.head(features)
            return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, return_probs: bool = True):
        """
        Inference mode prediction.

        Args:
            x: (B, C, H, W) input images
            return_probs: Return probabilities instead of logits

        Returns:
            predictions: Class predictions or probabilities
        """
        self.eval()
        logits = self.forward(x)

        if return_probs:
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            return logits

    def get_attention_maps(self, x: torch.Tensor, layer_idx: Optional[int] = None):
        """
        Extract attention maps for visualization.

        Args:
            x: (B, C, H, W) input image
            layer_idx: Specific layer index (None for all layers)

        Returns:
            Attention maps from specified layer(s)
        """
        self.eval()
        with torch.no_grad():
            _, attention_maps = self.forward(x, return_attention=True)

        if layer_idx is not None:
            return attention_maps[layer_idx]
        return attention_maps


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Args:
        alpha (float): Weighting factor [0, 1]
        gamma (float): Focusing parameter (typically 2.0)
        reduction (str): 'none', 'mean', or 'sum'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class labels

        Returns:
            loss: Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def build_vit_model(config: dict) -> VisionTransformer:
    """
    Factory function to build ViT model from config.

    Args:
        config: Configuration dictionary

    Returns:
        VisionTransformer model
    """
    model_cfg = config['model']

    model = VisionTransformer(
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size'],
        in_chans=model_cfg['in_channels'],
        num_classes=model_cfg['num_classes'],
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        mlp_ratio=model_cfg['mlp_ratio'],
        dropout=model_cfg.get('dropout', 0.1),
        attn_dropout=model_cfg.get('attn_dropout', 0.1)
    )

    return model


class ViTWithMAE(nn.Module):
    """
    Combined model: MAE for denoising + ViT for classification.

    This model can be used for end-to-end training or inference
    with a pre-trained MAE and ViT.

    Args:
        mae_model: Pretrained MAE model
        vit_model: ViT classification model
        freeze_mae: Whether to freeze MAE weights
    """
    def __init__(self, mae_model, vit_model, freeze_mae: bool = True):
        super().__init__()
        self.mae = mae_model
        self.vit = vit_model

        if freeze_mae:
            for param in self.mae.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, return_denoised: bool = False):
        """
        Forward pass through MAE then ViT.

        Args:
            x: (B, C, H, W) input images
            return_denoised: Whether to return denoised image

        Returns:
            logits: (B, num_classes) or (logits, denoised_img)
        """
        # Denoise with MAE
        with torch.no_grad() if not self.training else torch.enable_grad():
            denoised = self.mae.denoise(x)

        # Classify with ViT
        logits = self.vit(denoised)

        if return_denoised:
            return logits, denoised
        return logits


if __name__ == "__main__":
    # Test the model
    print("Testing Vision Transformer model...")

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.1
    )

    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Logits: {logits}")

    # Test with attention
    logits, attn_maps = model(x, return_attention=True)
    print(f"\nNumber of attention layers: {len(attn_maps)}")
    print(f"Attention map shape: {attn_maps[0].shape}")

    # Test prediction
    probs = model.predict(x)
    print(f"\nPrediction probabilities: {probs}")
    print(f"Predicted classes: {probs.argmax(dim=1)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test focal loss
    print("\nTesting Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.tensor([0, 1])
    loss = focal_loss(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")

    print("\nViT model test completed successfully!")
