import torch.nn as nn
from buildingblocks import Encoder, Decoder, DoubleConv
from utils import create_feature_maps
import torch
import torch.nn.functional as F
from einops import rearrange
from einops_exts import rearrange_many


def exists(x):
    return x is not None

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)
    
    
class SpectralAttentionWithRoPE(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x):  # x: (B, S, C)
        b, s, c = x.shape # 512 30 128

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, s, dim*3) (512, 30, 128)
        q, k, v = map(lambda t: rearrange(t, 'b s (h d) -> b h s d', h=self.heads), qkv) # (512, 4, 30, 32)

        q = q * self.scale
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) # (512, 4, 32, 32)

        attn = attn - attn.amax(dim = -1, keepdim = True).detach()

        attn = attn.softmax(dim = -1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v) # (512, 4, 30, 32)
        out = rearrange(out, 'b h s d -> b s (h d)') # (512, 30, 128)

        return self.to_out(out)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.spectral_attn = SpectralAttentionWithRoPE(dim, heads=heads, dim_head=dim_head)

    def forward(self, x):
        b, c, t, hw, s = x.shape # 1 128 2 256 30

        # reshape for spectral attention: treat each spatial location as independent spectral sequence
        x_spec = rearrange(x, 'b c t hw s -> (b t hw) s c') # (512, 30, 128)
        x_spec = self.spectral_attn(x_spec) # (512, 30, 128)
        x_spec = rearrange(x_spec, '(b t hw) s c -> b c t hw s', b=b, t=t, hw=hw) # (1, 128, 2, 256, 30)

        # Use spectral-attended features.
        x = x_spec  # (1, 128, 2, 256, 30)

        x = rearrange(x, 'b c t hw s -> (b t) c hw s') # (2, 128, 256, 30)

        qkv = self.to_qkv(x).chunk(3, dim = 1) # (2, 128, 256, 30)

        q, k, v = rearrange_many(qkv, 'b (he c) hw s -> b he (c s) hw', he = self.heads) # (2, 4, 960, 256)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v) # (2, 8, 960, 960)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q) # (2, 8, 960, 256)

        out = rearrange(out, 'b he (c s) hw -> b (he c) hw s', he = self.heads, s = s, hw = hw) # (2, 256, 256, 30)
        out = self.to_out(out) # (2, 256, 256, 30)
        return rearrange(out, '(b t) c hw s -> b c t hw s', b = b)


class ssan(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='ce', num_groups=8, 
                 heads = 2, use_sparse_linear_attn = True, **kwargs):
        super(ssan, self).__init__()

        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        encoders = []
        s_atts = []

        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups, pool_kernel_size=(2, 2, 2))
                s_att = Residual(PreNorm(out_feature_num, SpatialLinearAttention(out_feature_num, heads = heads, dim_head = 32))) if use_sparse_linear_attn else nn.Identity()

            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups, pool_kernel_size=(2, 2, 2))
                
                s_att = Residual(PreNorm(out_feature_num, SpatialLinearAttention(out_feature_num, heads = heads, dim_head = 32))) if use_sparse_linear_attn else nn.Identity()
                
            encoders.append(encoder)
            s_atts.append(s_att)
        
        self.encoders = nn.ModuleList(encoders)
        self.s_atts = nn.ModuleList(s_atts)

        decoders = []
        de_s_atts = []

        reversed_f_maps = list(reversed(f_maps))

        for i in range(len(reversed_f_maps) - 1):

            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]

            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups, scale_factor=(2, 2, 2))
            
            de_s_att = Residual(PreNorm(out_feature_num, SpatialLinearAttention(out_feature_num, heads = heads, dim_head = 32))) if use_sparse_linear_attn else nn.Identity()

            decoders.append(decoder)
            de_s_atts.append(de_s_att)

        self.decoders = nn.ModuleList(decoders)
        self.de_s_atts = nn.ModuleList(de_s_atts)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):

        encoders_features = []
        for encoder, s_att, in zip(self.encoders, self.s_atts):
            # (1, 2, 8, 1024, 120)
            x = encoder(x) # (1, 32, 8, 1024, 120) -> (1, 64, 4, 512, 60) -> (1, 128, 2, 256, 30) -> (1, 256, 1, 128, 15)

            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        for decoder, de_s_att, encoder_features in zip(self.decoders, self.de_s_atts, encoders_features):
            # (1, 256, 1, 128, 15)
            x = decoder(encoder_features, x) # (1, 128, 2, 256, 30) -> (1, 64, 4, 512, 60) -> (1, 32, 8, 1024, 120)

            x = de_s_att(x)

        x = self.final_conv(x) # (1, 2, 8, 1024, 120)

        return x