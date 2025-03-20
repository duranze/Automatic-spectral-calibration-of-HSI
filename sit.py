import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import os
sys.path.append(os.getcwd())

# Utility functions for reshaping tensors
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# Bias-free Layer Normalization
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

# Layer Normalization with bias
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# Wrapper for Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Multi-Scale Feed-Forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden_features * 3, kernel_size=(1, 1, 1), bias=bias)
        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3, 3, 3),
                                  stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3),
                                  stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3),
                                  stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1, 1, 1), bias=bias)
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        x = F.gelu(x1) * x2 * x3
        x = x.unsqueeze(2)
        x = self.project_out(x)
        return x.squeeze(2)

# Convolution and Attention Fusion Module (CAFM)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3),
                                    stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3),
                                  groups=dim // self.num_heads, padding=1, bias=True)
        self.l2q = nn.Linear(dim, dim)
    def forward(self, x, x_d):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x)).squeeze(2)
        f_all = qkv.reshape(b, h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(b, 9 * c // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv).squeeze(2)
        q_0 = self.l2q(x_d)
        q_0 = rearrange(q_0, 'b (head c) -> b head c', head=self.num_heads)
        q_0 = torch.matmul(q_0.unsqueeze(-1), q_0.unsqueeze(-2))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn * q_0
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out).squeeze(2)
        return out + out_conv

# CAMixing Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, avg_nums):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.conv2avg = Conv2DTransformer(dim, LayerNorm_type, avg_nums)
    def forward(self, x):
        x_d = self.conv2avg(x)
        x_d = F.gelu(x_d)
        x_att = self.attn(self.norm1(x), x_d)
        x = x + x_att
        x = x + self.ffn(self.norm2(x))
        return x

# Dividing normalization
class DivNormalization(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(DivNormalization, self).__init__()
        self.epsilon = epsilon
    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + self.epsilon)

# Convolutional Transformer for downscaling features
class Conv2DTransformer(nn.Module):
    def __init__(self, in_channels, LayerNorm_type, num_conv_layers=3):
        super(Conv2DTransformer, self).__init__()
        layers = []
        current_channels = in_channels
        # Define a series of convolution and pooling layers
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1))
            layers.append(LayerNorm(in_channels, LayerNorm_type))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
        self.conv_layers = nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), x.size(1), -1)
        return torch.mean(x, dim=-1)

# Overlap Patch Embedding using 3D convolution
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=31, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=(3, 3, 3), stride=1, padding=1, bias=bias)
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.proj(x)
        return x.squeeze(2)

# Downsampling module
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x):
        return self.body(x)

# Upsampling module
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.body(x)

# The main SIT network
class SIT(nn.Module):
    def __init__(self, inp_channels=204, out_channels=204, dim=48,
                 num_blocks=[2, 3, 3, 4], num_refinement_blocks=1,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super(SIT, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type, avg_nums=3)
                                               for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2), num_heads=heads[1],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type, avg_nums=2)
                                               for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 4), num_heads=heads[2],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type, avg_nums=1)
                                               for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv3d(int(dim * 4), int(dim * 2), kernel_size=(1, 1, 1), bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2), num_heads=heads[1],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type, avg_nums=1)
                                               for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2), num_heads=heads[0],
                                                               ffn_expansion_factor=ffn_expansion_factor,
                                                               bias=bias, LayerNorm_type=LayerNorm_type, avg_nums=2)
                                               for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim * 2), num_heads=heads[0],
                                                            ffn_expansion_factor=ffn_expansion_factor,
                                                            bias=bias, LayerNorm_type=LayerNorm_type, avg_nums=3)
                                            for _ in range(num_refinement_blocks)])
        self.output = nn.Conv3d(int(dim * 2), out_channels, kernel_size=(3, 3, 3),
                                stride=1, padding=1, bias=bias)
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_dec_level3 = out_enc_level3
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2.unsqueeze(2)).squeeze(2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1.unsqueeze(2)).squeeze(2) + inp_img
        return out_dec_level1
