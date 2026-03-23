import torch
import torch.nn as nn
import timm
import open_clip
import numpy as np
from torch.utils.checkpoint import checkpoint
import torchvision.models
import torch.nn.functional as F
# import vit_pytorch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.vit import Transformer
from models.swin_transformer import SwinTransformer3D
#from models.head import VQAHead
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

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x  
    
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


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=384, hidden_channels=64, out_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, self.out_channels, (1, 1, 1))
        self.gelu = nn.GELU()

        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x, rois=None):
        # x is [B, C, T, H, W] -> [16, 768, 16, 7, 7]
        x = self.dropout(x)
        pred =self.pool(self.fc_last(self.dropout(self.gelu(self.fc_hid(x)))))
        return pred


class pre_trained_model(nn.Module):
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
                 use_checkpoint=False
                 ):

        super().__init__()
        
        self.pos_margin = 0.05
        self.transformer3d = SwinTransformer3D(depths=depths, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate)
        
        self.linear = nn.Linear(768, 60)
        self.cross_attention = CrossAttention(feature_dim=60, num_heads=1)
        
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.pool3d = nn.AdaptiveAvgPool3d(1)
        
        self.sobel_x, self.sobel_y = get_sobel(768, 1)
        self.erb = ERB(768,128)

        self.pred = MLP(1076,128,1)
        self.dis_type = MLP(956,128,10)
        self.dis_degree = MLP(956,128,3)
        

    def forward(self, ref,ref_rank,ref_rank_A,ref_rank_T,ref_T):
        batch, dis_level, C1, D1, H1, W1 = ref_rank.size()

        pos_dis=[]
        ref_rank = ref_rank.contiguous().view(batch*dis_level,C1,D1,H1,W1)#[B * dis_leve, C, D, H, W]

        ref_rank = self.transformer3d(ref_rank)#[batchsize * dis_leve, 768, D, 7, 7]
        ref = self.transformer3d(ref)
        B, C, D, H, W = ref_rank.size()
        
        
        ref_rank_pool = self.pool3d(ref_rank)#[batchsize * dis_leve, D, 768, 1, 1]
        ref_pool = self.pool3d(ref)
        ref_rank_pool = ref_rank_pool.contiguous().view(B, C)
        ref_pool = ref_pool.contiguous().view(batch, C)

        ref_rank = ref_rank.contiguous().view(B * D, C, H, W)
        ref = ref.contiguous().view(batch * D, C, H, W)
        
        #extract edge and fusion
        ref_rank_edge = self.erb(run_sobel(self.sobel_x, self.sobel_y, ref_rank))
        ref_edge = self.erb(run_sobel(self.sobel_x, self.sobel_y, ref))
        ref_rank = torch.cat((ref_rank,ref_rank_edge), dim=1)
        ref = torch.cat((ref,ref_edge), dim=1)
        ref_rank = ref_rank.contiguous().view(B, 896, D, H, W)
        ref = ref.contiguous().view(batch, 896, D, H, W)
        

        
        #lbp features
        ref_rank_linear = self.linear(ref_rank_pool)
        ref_linear = self.linear(ref_pool)
        ref_rank_cross_T = self.cross_attention(ref_rank_T,ref_rank_linear)
        ref_cross_T = self.cross_attention(ref_T,ref_linear)
        
        #lbp  edge fusion
        ref_rank = self.pool3d(ref_rank)#[batchsize * dis_leve, D, 896, 1, 1]
        ref = self.pool3d(ref)
        ref_rank = ref_rank.contiguous().view(B, 896)
        ref = ref.contiguous().view(batch, 896)
        ref_rank = torch.cat((ref_rank,ref_rank_cross_T), dim=1)
        ref = torch.cat((ref,ref_cross_T), dim=1)
        
        ref_rank = torch.cat((ref_rank,ref_rank_A), dim=1)


        dis_rank = self.pred(ref_rank)
        dis_type = self.dis_type(ref)
        dis_degree = self.dis_degree(ref)

        
        dis_type = dis_type.squeeze()
        dis_degree = dis_degree.squeeze()
        
        #print(scores)
        for i in range(batch):
            for j in range(dis_level*i,dis_level*(i+1)):
                 for k in range(j+1,dis_level*(i+1)):
                     pos_dis.append(dis_rank[k][0]-dis_rank[j][0])
                
        pos_dis = torch.stack(pos_dis,dim=0)   
        pos_loss = torch.maximum(torch.zeros_like(pos_dis),self.pos_margin+pos_dis)
        loss_pos = torch.sum(pos_loss)/batch
        #print("loss_neg",loss_neg)
       ## print("loss_pos",loss_pos)
       
            
        
        
        return loss_pos,dis_type,dis_degree
        
        
def nt_xent_loss(a: torch.Tensor, b: torch.Tensor, tau: float = 0.1):
    """
    Compute the NT-Xent loss.

    Args:
        a (torch.Tensor): first set of features
        b (torch.Tensor): second set of features
        tau (float): temperature parameter
    """
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)        


