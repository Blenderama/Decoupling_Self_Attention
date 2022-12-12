""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from einops import rearrange
from modules.layers_ours import *

from baselines.ViT.helpers import load_pretrained
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.layer_helpers import to_2tuple
import numpy as np
import math
import pdb
import time

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

def compute_rollout_attention_plus(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices)-1)] + [all_layer_matrices[-1]]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]

    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x, keeptoken=None, base_attn=None, ext_attn=None, ext_v=None, ext_grad=None):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        if ext_v is not None:
            v = ext_v
        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale
        
        if ext_attn is None:
            attn = self.softmax(dots)
            attn = self.attn_drop(attn)
        else:
            attn = ext_attn

        self.save_attn(attn)
        
        if ext_grad is None:
            base_coef = 1
        else:
#             base_coef = (ext_grad.mean(1, keepdims=True).repeat(1,12, 1,1) > 0) * 1.
            base_coef = (ext_grad > 0) * 1.
        if keeptoken is None:
            try:
                attn.register_hook(self.save_attn_gradients)
            except:
                pass
        else:
            attn_ = attn.clone() * 0
            attn_[..., keeptoken] = (attn * base_coef)[..., keeptoken]
            attn = attn_

#             attn_ = attn.clone()
#             attn_[..., keeptoken] = 0
#             attn = attn_
        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def fast_forward(self, res_input, base_attn=None, ext_attn=None, ext_v=None, ext_grad=None):
        b, n, _, h = *res_input.shape, self.num_heads
        if ext_grad is None:
            base_coef = 1
        else:
            base_coef = (ext_grad > 0) * 1.
            
#         attn_v = []
#         for i in range(1, ext_v.shape[2]):
#             out = ext_attn[..., i].unsqueeze(-1) @ ext_v[..., i, :].unsqueeze(-2)
#             out = rearrange(out, 'b h n d -> b n (h d)')
#             out = self.proj(out)
#             attn_v.append(out)
#         attn_v = torch.cat(attn_v)
        attn_v = ext_attn[0, ..., 1:].permute(-1, 0, 1).unsqueeze(-1) * ext_v[0, :, 1:, :].permute(1, 0, 2).unsqueeze(-2)
        attn_v = rearrange(attn_v, 'b h n d -> b n (h d)')
        attn_v = self.proj(attn_v)
        
        return attn_v

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def get_y(self):
        return self.y

    def save_y(self, y):
        self.y = y
        
    def get_x1(self):
        return self.x1

    def save_x1(self, x1):
        self.x1 = x1
        
    def get_av(self):
        return self.av

    def save_av(self, av):
        self.av = av
        
    def forward(self, x, keeptoken=None, base_attn=None, ext_attn=None, ext_v=None, ext_y=None, ext_x1=None, ext_grad=None):
        x1, x2 = self.clone1(x, 2)
        
        av = self.attn(self.norm1(x2), keeptoken, base_attn, ext_attn, ext_v, ext_grad)
        x = self.add1([x1, av])
#         else:
#             av = self.attn(self.norm1(x2), keeptoken, base_attn, ext_attn, ext_v)
#             x = self.add1([ext_x1, av])
        self.save_av(av)
        x1, x2 = self.clone2(x, 2)
        self.save_x1(x1)
        if ext_y is None:
            y = self.mlp(self.norm2(x2))
            self.save_y(y)
        else:
            y = ext_y
        if ext_x1 is not None:
            x1 = ext_x1
        x = self.add2([x1, y])
        return x

    def fast_forward(self, res_input, base_attn=None, ext_attn=None, ext_v=None, ext_y=None, ext_x1=None, ext_grad=None):
        attn_v = self.attn.fast_forward(res_input, base_attn, ext_attn, ext_v, ext_grad)
        res_input = res_input + attn_v + ext_y
        return res_input

#     def fast_forward(self, res_input, base_attn=None, ext_attn=None, ext_v=None, ext_y=None, ext_x1=None, ext_grad=None):
#         attn_v = self.attn.fast_forward(res_input, base_attn, ext_attn, ext_v, ext_grad)
#         res_input = self.norm2(res_input + attn_v) + ext_y
#         return self.norm1(res_input)
    
    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1,2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        self.norm = LayerNorm(embed_dim)
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = Linear(embed_dim, num_classes)

        # FIXME not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None
        
        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, keeptoken=None, base_attn=None, ext_attn_list=None, ext_v_list=None, start_layer=None, ext_y_list=None, ext_x1_list=None, ext_grad_list=None, keep_patch=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])
        if keep_patch is not None:
            x_ = x * 0
            x_[:, keep_patch, :] = x[:, keep_patch, :]
            x = x_
        if keeptoken is None:
            try:
                x.register_hook(self.save_inp_grad)
            except:
                pass

        for i, blk in enumerate(self.blocks):
#             x = blk(x, keeptoken)
#             x = blk(x)
            if ext_attn_list is None:
                ext_attn = None
            else:
                ext_attn = ext_attn_list[i]
            if ext_v_list is not None:
                ext_v = ext_v_list[i]
            else:
                ext_v = None
            if ext_y_list is None:
                ext_y = None
            else:
                ext_y = ext_y_list[i]
            if ext_x1_list is None:
                ext_x1 = None
            else:
                ext_x1 = ext_x1_list[i]
            if ext_grad_list is None:
                ext_grad = None
            else:
                ext_grad = ext_grad_list[i]
#             if i == len(self.blocks) - 1:

#             x = blk(x)

            if start_layer is None:
                x = blk(x, ext_y=ext_y)
            else:
                if i >= start_layer:
                    x = blk(x, keeptoken, base_attn, ext_attn, ext_v, ext_y=ext_y, ext_x1=ext_x1, ext_grad=ext_grad)

#                     if i == 11:
#                         x = blk(x, keeptoken, base_attn, ext_attn, ext_v, ext_y=ext_y, ext_x1=ext_x1, ext_grad=ext_grad)
#                     else:
#                         x = blk(x, keeptoken, base_attn, ext_attn, ext_v, ext_y=ext_y, ext_x1=None, ext_grad=ext_grad)
                else:
                    x = blk(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def fast_forward(self, x, base_attn=None, ext_attn_list=None, ext_v_list=None, start_layer=0, ext_y_list=None, ext_x1_list=None, ext_grad_list=None):
#         attended_v = []
#         residual_input = []
        x = self.patch_embed(x)
        
        x_ = torch.zeros_like(x.repeat(x.shape[1], 1, 1))
        B = x_.shape[0]
        for i in range(x_.shape[0]):
            x_[i, i, :] = x[0, i, :]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        res_input = torch.cat((cls_tokens, x_), dim=1)
        res_input = self.add([res_input, self.pos_embed])
        
        for i, blk in enumerate(self.blocks):
            if ext_attn_list is None:
                ext_attn = None
            else:
                ext_attn = ext_attn_list[i]
            if ext_v_list is not None:
                ext_v = ext_v_list[i]
            else:
                ext_v = None
            if ext_y_list is None:
                ext_y = None
            else:
                ext_y = ext_y_list[i]
            if ext_x1_list is None:
                ext_x1 = None
            else:
                ext_x1 = ext_x1_list[i]
            if ext_grad_list is None:
                ext_grad = None
            else:
                ext_grad = ext_grad_list[i]
            
            if i >= start_layer:
                res_input = blk.fast_forward(res_input, base_attn, ext_attn, ext_v, ext_y=ext_y, ext_x1=ext_x1, ext_grad=ext_grad)
            else:
                x = blk(x)
                res_input = x.repeat(x.shape[1] - 1, 1, 1)
                
        res_input = self.norm(res_input)
        res_input = self.pool(res_input, dim=1, indices=torch.tensor(0, device=x.device))
        res_input = res_input.squeeze(1)
        res_input = self.head(res_input)
        return res_input

    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, grads_all=None, nth_grads=None, x0=None, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            time_now = time.time()
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
#                 grad = self.blocks[-1].attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
#                 cam = blk.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))

            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam#, time.time() - time_now
        
        elif method == "sas":
            cams = 0
            time_now = time.time()
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams += cam

            cam = cams[0, 1:]
            return cam#, time.time() - time_now
        
        elif method == "taylor":
#             grad = nth_grads[0][0].clamp(0).mean(0)[0, 1:]

            grad = self.blocks[-1].attn.get_attn_gradients()[0].clamp(0).mean(0)[0, 1:]
#             grad = self.blocks[-1].attn.get_attn()[0].clamp(0).mean(0)[0, 1:]
            cam  = self.blocks[-1].attn.get_attn()[0].clamp(0).mean(0)[0, 1:]
            cam = grad * cam
            return cam
    
        elif method == "taylor_2nd_order":
            grad_1 = nth_grads[0].clamp(0).mean(0)
            grad_2 = nth_grads[1].clamp(0).mean(0)
            cam = self.blocks[-1].attn.get_attn()[0].clamp(0).mean(0)[0, 1:]

            cam = grad_1 * cam - 0.5 * (cam**2) * grad_2
            return cam
    
        elif method == "taylor_cam":
            cams = 0
            for i in range(len(nth_grads)):
                grad = nth_grads[i][0]

                cam = ((self.blocks[-1].attn.get_attn()) ** (i+1))[0]
                cam = grad * cam * (1 / math.factorial(i+1))

                cams += cam
            cams = cams.clamp(min=0).mean(dim=0)
#             cams = cams.mean(dim=0)

            cam = cams[0, 1:]
            return cam
        
        elif method == "transformer_attribution_last":
            cams = []
            for blk in self.blocks:
#                 grad = blk.attn.get_attn_gradients()
                grad = self.blocks[-1].attn.get_attn_gradients()
#                 cam = blk.attn.get_attn_cam()
                cam = blk.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)

            cam = rollout[:, 0, 1:]
            return cam
        
        elif method == "transformer_attribution_both_last":
            cams = []
            for blk in self.blocks:
#                 grad = blk.attn.get_attn_gradients()
#                 cam = blk.attn.get_attn_cam()
                grad = self.blocks[-1].attn.get_attn_gradients()
                cam = self.blocks[-1].attn.get_attn_cam()
#                 cam = blk.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)

            cam = rollout[:, 0, 1:]
            return cam
        
        elif method == "transformer_attribution_cam_first":
            cams = []
            for blk in self.blocks:
#                 grad = blk.attn.get_attn_gradients()
#                 cam = blk.attn.get_attn_cam()
                grad = self.blocks[-1].attn.get_attn_gradients()
                cam = self.blocks[start_layer].attn.get_attn_cam()
#                 cam = blk.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)

            cam = rollout[:, 0, 1:]
            return cam
        
        elif method == "transformer_attribution_2nd_last":
            cams = []
            for blk in self.blocks:
#                 grad = blk.attn.get_attn_gradients()
                grad = self.blocks[-2].attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
#                 cam = blk.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)

            cam = rollout[:, 0, 1:]
            return cam
        
        elif method == "transformer_attribution_plus":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention_plus(cams, start_layer=start_layer)

            cam = rollout[:, 0, 1:]
            return cam
        
        elif method == "t_cam_diff":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            
            num_tokens = self.blocks[-1].attn.get_attn_cam().shape[1]
            batch_size = self.blocks[-1].attn.get_attn_cam().shape[0]
            t_cams = 0 #torch.zeros(num_tokens).expand(batch_size, num_tokens, num_tokens).to(cam.device)
            for i, blk in enumerate(self.blocks):
                if i <= start_layer:
                    continue
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad.clamp(min=0) * cam
#                 cams += cam * (0.5 ** (len(self.blocks) - i - 1))
                t_cams += cam
            t_cams = t_cams.clamp(min=0).mean(dim=0)
            assert((rollout[:, 0, 1:] - t_cams[0, 1:]).min()) >= 0
            return rollout[:, 0, 1:] - t_cams[0, 1:]
        
        elif method == "transformer_attribution_markov":
            b, h, s, _ = self.model.blocks[-1].attn.get_attn_cam().shape
            states = self.model.blocks[-1].attn.get_attn_cam().mean(1)[:, 0, :].reshape(b, 1, s)
            for blk in self.blocks[:-1]:
#                 grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#                 grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
#                 cam = grad * cam
                cam = cam.mean(dim=0)
#                 cams.append(cam.unsqueeze(0))
                states += states @ cam
            states = states * self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
            return states[:, 0, 1:]
        
        elif method == "transformer_attribution_dot":
            num_tokens = self.blocks[-1].attn.get_attn_cam().shape[-1]
            batch_size = self.blocks[-1].attn.get_attn_cam().shape[0]
            cams = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(cam.device)
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams += cams @ cam.unsqueeze(0)
#             rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = cams[:, 0, 1:]
            return cam
        
        elif method == "igrad": # layer-wise and head-wise combinatioin
#             cams = 0
#             for i, blk in enumerate(self.blocks):
#                 if i <= start_layer:
#                     continue
#                 grad = grads_all[i]
# #                 cam = blk.attn.get_attn_cam()
#                 cam = blk.attn.get_attn()
#                 cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#                 cam = grad.clamp(min=0) * cam    
#                 cams += cam
#             cams = cams.clamp(min=0).mean(dim=0)
#             cams = compute_rollout_attention_plus(cams, start_layer=start_layer)
            cams = []
            for i, blk in enumerate(self.blocks):
#                 cam = blk.attn.get_attn_cam()
                cam = blk.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grads_all[i]
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention_plus(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]

            return cam
        
        elif method == "igrad2": # layer-wise and head-wise combinatioin
            grads_all = 0
            for alpha in np.linspace(0, 1, 20):
                grads = []
                for blk in self.blocks:
                    grad = blk.attn.get_attn_gradients()
#                     cam = blk.attn.get_attn_cam()
#                     cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                    grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                    grads.append(grad)
                grads_all += torch.stack(grads)
            cams = []
            for i, blk in enumerate(self.blocks):
                grad = grads_all[i]
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                cam = grad.clamp(min=0) * cam
                cam = cam.mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam
        
        elif method == "igrad3": # layer-wise and head-wise combinatioin
            grads_all = 0
            num_tokens = self.blocks[-1].attn.get_attn_cam().shape[-1]
            batch_size = self.blocks[-1].attn.get_attn_cam().shape[0]
            cams = torch.eye(num_tokens).expand(12, num_tokens, num_tokens).to(cam.device)
            for alpha in np.linspace(0, 1, 20):
                grads = []
                for blk in self.blocks:
                    grad = blk.attn.get_attn_gradients()
                    cam = blk.attn.get_attn_cam()
                    cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                    grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                    grads.append(grad)
                grads_all += torch.stack(grads)
#             cams = []
            for i, blk in enumerate(self.blocks):
                grad = grads_all[i]
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                cam = grad * cam
#                 cam = cam.clamp(min=0).mean(dim=0)
#                 cams.append(cam.unsqueeze(0))
                cams += cams @ cam

            cams = cams.clamp(min=0).mean(dim=0)
#             rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = cams[0, 1:]
            return cam
        
        elif method == "t_cam":
            num_tokens = self.blocks[-1].attn.get_attn_cam().shape[1]
            batch_size = self.blocks[-1].attn.get_attn_cam().shape[0]
            cams = 0 #torch.zeros(num_tokens).expand(batch_size, num_tokens, num_tokens).to(cam.device)
            for i, blk in enumerate(self.blocks):
                if i <= start_layer:
                    continue
#                 grad = blk.attn.get_attn_gradients()
                grad = self.blocks[-1].attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
#                 cam = blk.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad.clamp(min=0) * cam
#                 cams += cam * (0.5 ** (len(self.blocks) - i - 1))
                cams += cam# * max(13 - i, 0)
#                 cams += cam
            cams = cams.clamp(min=0).mean(dim=0)
            return cams[0, 1:]

        elif method == "tam":
            num_tokens = self.blocks[-1].attn.get_attn_cam().shape[1]
            batch_size = self.blocks[-1].attn.get_attn_cam().shape[0]
            cams = 0 #torch.zeros(num_tokens).expand(batch_size, num_tokens, num_tokens).to(cam.device)
            for i, blk in enumerate(self.blocks):
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad.clamp(min=0) * cam
                cams += cam * (0.5 ** (len(self.blocks) - i - 1))

            cams = cams.clamp(min=0).mean(dim=0)
#             rollout = compute_rollout_attention(cams, start_layer=start_layer)
#             cam = rollout[:, 0, 1:]
            return cams[0, 1:]
            
        elif method == "t_cam_grad":
            num_tokens = self.blocks[-1].attn.get_attn_cam().shape[1]
            batch_size = self.blocks[-1].attn.get_attn_cam().shape[0]
            cams = 0 #torch.zeros(num_tokens).expand(batch_size, num_tokens, num_tokens).to(cam.device)
            for i, blk in enumerate(self.blocks):
                if i <= start_layer:
                    continue
#                 grad = blk.attn.get_attn_gradients()
                grad = self.blocks[-1].attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
#                 cam = blk.attn.get_attn()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = self.gap(grad * cam / cam.sum()).clamp(min=0) * cam
#                 cam = self.gap(grad * cam / cam.sum()) * cam
                cams += cam
            cams = cams.clamp(min=0).mean(dim=0)
            return cams[0, 1:]

        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model