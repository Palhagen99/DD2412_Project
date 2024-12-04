# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed
from timm.layers import Mlp
from pos_embed import get_2d_sincos_pos_embed

import numpy as np
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm=qk_norm
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_drop=proj_drop
        self.norm_layer=norm_layer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        
    # We need custom attention in blocks
    # Regular attention for encoder
    def attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3,
                                self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        #q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm=qk_norm
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_drop=proj_drop
        self.norm_layer=norm_layer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        
    # We need custom attention in blocks
    # Cross attention for decoder
    def cross_attn(self, x1, x2):
        B, N, C = x1.shape
        qkv1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1.unbind(0)

        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2.unbind(0)

        x = F.scaled_dot_product_attention(q2, k1, v1)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

    def attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

    def forward(self, x1, x2, cross_self=True, cross=False, joint=False):
        if cross_self:
            x = x2 + self.cross_attn(self.norm1(x1), self.norm1(x2))
            x = x + self.attn(self.norm2(x))
        elif cross:
            x = x2 + self.cross_attn(self.norm1(x1), self.norm1(x2))
        
        if joint:
            x = x1 + self.mlp(self.norm3(x1))
        else:
            x = x + self.mlp(self.norm3(x))
            
        return x

class SiameseMaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, use_joint_enc=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        use_joint_enc: use the joint encoder method
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # -----------------------------------------------------------------
        if use_joint_enc:
            bmask = torch.zeros_like(mask, dtype=torch.bool)
            x_masked = x.clone()
            x_masked[~bmask] = 0 
        # -----------------------------------------------------------------

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, use_joint_enc=False, use_joint_dec=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # -----------------------------------------------------------------
        if mask_ratio>0 or use_joint_enc or use_joint_dec:
            if use_joint_enc:
                x1_idx = int(x.shape[0] / 2)
                x2_idx = x.shape[0]
                x1, mask1, ids_restore1 = self.random_masking(x[:x1_idx], mask_ratio=0)
                x2, mask2, ids_restore2 = self.random_masking(x[x1_idx:x2_idx], mask_ratio=mask_ratio)
                x = torch.vstack([x1, x2])
                mask = torch.vstack([mask1, mask2])
                ids_restore = torch.vstack([ids_restore1, ids_restore2])
            else: 
                # masking: length -> length * mask_ratio
                x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # -----------------------------------------------------------------
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
       
        # -----------------------------------------------------------------
        if mask_ratio>0 or use_joint_enc or use_joint_dec:
            return x, mask, ids_restore
        else:
           return x
        # -----------------------------------------------------------------


     # ----------------------------------------------------------------- 
    def forward_decoder(self, f1, f2, ids_restore_2, use_joint_enc=False):

        x_1 = self.decoder_embed(f1)
        x_1 = x_1 + self.decoder_pos_embed

        x_2 = self.decoder_embed(f2)
        mask_tokens = self.mask_token.repeat(f2.shape[0], ids_restore_2.shape[1] + 1 - x_2.shape[1], 1)
        x_2_ = torch.cat([x_2[:, 1:, :], mask_tokens], dim=1)
        x_2_ = torch.gather(x_2_, dim=1, index=ids_restore_2.unsqueeze(-1).repeat(1, 1, x_2.shape[2]))
        x_2 = torch.cat([x_2[:, :1, :], x_2_], dim=1)

        x_2 = x_2 + self.decoder_pos_embed

        if use_joint_enc:
          x_1 = torch.vstack([x_1, x_2])

        for blk in self.decoder_blocks:
            x_2 = blk(x_1, x_2)

        x = self.decoder_norm(x_2)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x
    # -----------------------------------------------------------------



    def forward_loss(self, imgs, pred, mask, use_joint_dec=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        
        # -----------------------------------------------------------------
        if use_joint_dec:
          target = self.patchify(imgs[:, :, :, :])
        else:
          target = self.patchify(imgs[:, 1, :, :, :])
        # -----------------------------------------------------------------

        
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.95, use_joint_enc=False, use_joint_dec=False):
        if use_joint_enc:
          imgs_joint = torch.vstack([imgs[:, 0], imgs[:, 1]])
          latent, mask, ids_restore = self.forward_encoder(imgs_joint.float(), mask_ratio=mask_ratio)
          l1_idx = int(latent.shape[0] / 2)
          l2_idx = latent.shape[0]
          latent_1 = latent[:l1_idx]
          latent_2 = latent[l1_idx:l2_idx]
          ids_restore_2 = ids_restore[l1_idx:l2_idx]
          mask_1 = mask[:l1_idx]
          mask_2 = mask[l1_idx:l2_idx]
        elif use_joint_dec:
          latent_1, mask_1, _ = self.forward_encoder(imgs[:, 0].float(), mask_ratio=0)
          latent_2, mask_2, ids_restore_2 = self.forward_encoder(imgs[:, 1].float(), mask_ratio=mask_ratio)
        else:
          latent_1 = self.forward_encoder(imgs[:, 0].float(), mask_ratio=0)
          latent_2, mask_2, ids_restore_2 = self.forward_encoder(imgs[:, 1].float(), mask_ratio=mask_ratio)

        if use_joint_dec:
          imgs = torch.vstack([imgs[:, 0], imgs[:, 1]])
          mask_2 = torch.vstack([mask_1, mask_2])
        pred = self.forward_decoder(latent_1, latent_2, ids_restore_2)
        loss = self.forward_loss(imgs, pred, mask_2)
        return loss, pred


def sim_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = SiameseMaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




# set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
