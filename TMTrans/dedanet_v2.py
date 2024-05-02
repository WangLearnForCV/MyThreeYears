import warnings
import torch.nn.functional as F
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_
import math
from timm.models.layers import DropPath
from torch.nn import Module
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn
import torch
from blocks import simam

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        #print('mlp run ...')
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear=nn.Linear(1024,4)
        self.conv1x1 = nn.Conv2d(896, 512,kernel_size=1)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W,xg):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        qg = self.q(xg).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        attn_g = (qg @ k.transpose(-2, -1)) * self.scale
        attn_g = attn_g.softmax(dim=-1)
        attn_g = self.attn_drop(attn_g)

        xg = (attn_g @ v).transpose(1, 2).reshape(B, N, C)
        xg = self.proj(xg)
        xg = self.proj_drop(xg)

        return x,xg


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dim=dim

    def forward(self, x, H, W,xg):
        #print('Block H W ',H,W,x.shape,self.dim)
        att,attg=self.attn(self.norm1(x), H, W,self.norm1(xg))
        x = x + self.drop_path(att)
        #print(x.shape,'  block x shape 1')
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        xg = self.drop_path(attg)
        # print(x.shape,'  block x shape 1')
        xg = xg + self.drop_path(self.mlp(self.norm2(xg), H, W))

        return x,xg

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=5, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.embeddim=embed_dim

    def forward(self, x):
        x = self.proj(x)
        #print('overlap patch embed run ....',self.embeddim)
        _, _, H, W = x.shape
        #print('overlap patch embed H W',H,W)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        #print('overlap  out  x  shape  ',x.shape)
        return x, H, W

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, floor=0,num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3], sr_ratios=[8, 4, 2, 1],in_val=None,in_g=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=5, stride=2, in_chans=in_chans,
                                              embed_dim=embed_dims[floor])
        # patch_embed
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size, patch_size=5, stride=2, in_chans=in_g,
                                              embed_dim=embed_dims[floor])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[floor], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.norm1 = norm_layer(embed_dims[floor])
        self.conv1x1 = nn.Conv2d(embed_dims[floor]*2,embed_dims[floor],kernel_size=1,padding=0)
        self.simam = simam()

    def forward_features(self, x,glcm):
        B = x.shape[0]
        #print('MixVisionTransformer run ....',x.shape)
        # stage 1
        x, H, W = self.patch_embed1(x)
        xg, Hg, Wg = self.patch_embed2(glcm)
        #print(x.shape,'   ---   ',xg.shape)
        #print('MixVisionTransformer   H W',H,W)
        for i, blk in enumerate(self.block1):
            x,xg = blk(x, H, W,xg)
        #print(x.shape,'   after patch_embed')
        #print(v.shape,'  this is the block 3  v shape')
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xg = self.norm1(xg)
        xg = xg.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print(x.shape,'   after reshape   ' , xg.shape)
        xg = self.simam(xg)
        x = torch.cat([x,xg],dim=1)
        x = self.conv1x1(x)
        return x

    def forward(self, x,glcm):
        x= self.forward_features(x,glcm)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        #print('DWConv ...')
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

        
if __name__=='__main__':
    img = torch.randn(2, 3, 256, 256)
    glcm = torch.randn(2, 1, 256, 256)
    t= MixVisionTransformer(in_chans=3,floor=0,depths=[3])

    preds = t(img, glcm)  # (1,1000)
    print('pre shape           ')
    print(preds.shape)

