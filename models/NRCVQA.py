import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import open_clip
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint
import torchvision.models
import torch.nn.functional as F
# import vit_pytorch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.swin_transformer import SwinTransformer3D
from models.vit import Transformer

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.attn_mask = clip_model.attn_mask

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input
    
class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x+res

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_ln = nn.Linear(in_channels, hidden_channels, bias=False)
        self.out_ln = nn.Linear(hidden_channels, out_channels, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, x):
        return self.out_ln(self.dropout(self.gelu(self.in_ln(x))))

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        self.query_linear = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_linear = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_linear = nn.Linear(feature_dim, feature_dim, bias=False)



    def forward(self, feature_A, feature_B):
        # Compute queries, keys, values
        queries = self.query_linear(feature_A).view(-1, self.num_heads, self.head_dim)
        keys = self.key_linear(feature_B).view(-1, self.num_heads, self.head_dim)
        values = self.value_linear(feature_B).view(-1, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        attention_scores = torch.einsum("bhd,bhd->bh", queries, keys) / (self.head_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        attended_features = attention * values

        # Combine heads
        attended_features = attended_features.reshape(-1, self.feature_dim)

        return attended_features




class NRCVQA(nn.Module):
    def __init__(self, text_tokens_T,
                 text_tokens_A,
                 text_tokens_C,
                 embedding_T,
                 embedding_A,
                 embedding_C,
                 text_encoder,
                 n_ctx,
                 img_size=14,
                 patch_size=(2,4,4),
                 in_chans=1024,
                 num_classes=1,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(8,7,7),
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 ape=False,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 attn_drop_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,):

        super().__init__()
        self.transformer3d = SwinTransformer3D(depths=depths, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate)
        #self.gelu = nn.GELU()
        self.linear = nn.Linear(768, 60)
        self.cross_attention = CrossAttention(feature_dim=60, num_heads=1)
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.pool3d = nn.AdaptiveAvgPool3d(1)
        
        self.sobel_x, self.sobel_y = get_sobel(768, 1)
        self.erb = ERB(768,128)
        
        self.pred = MLP(1076,128,1)

        
    def forward(self, x,y,z):
        x = self.transformer3d(x)
        # B D C H W
        B, C, D, H, W = x.size()
        x_pool = self.pool3d(x)
        
        #extract edge and contact
        x = x.contiguous().view(B * D, C, H, W)
        x_edge = self.erb(run_sobel(self.sobel_x, self.sobel_y, x))
        x = torch.cat((x,x_edge), dim=1)
        x = x.contiguous().view(B, 896, D, H, W)
        
        #cross attention lbp
        
        x_pool = x_pool.contiguous().view(B, C)
        x_linear = self.linear(x_pool)
        x_cross_T = self.cross_attention(z,x_linear)
        
        #contract
        x = self.pool3d(x)
        x = x.contiguous().view(B, 896)
        x = torch.cat((x,x_cross_T), dim=1)
        x = torch.cat((x,y), dim=1)
        
        scores = self.pred(x)
        

        
        return scores
