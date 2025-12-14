import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from Embedding import DataEmbedding
from entmax import entmax15

class Temporal_Attention_layer(nn.Module):
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.randn(num_of_vertices).to(device))  #35
        self.U2 = nn.Parameter(torch.randn(in_channels, num_of_vertices).to(device))  #[6,35]
        self.U3 = nn.Parameter(torch.randn(in_channels).to(device))  #[6]
        self.be = nn.Parameter(torch.randn(1, num_of_timesteps, num_of_timesteps).to(device))  #[1,1,1]
        self.Ve = nn.Parameter(torch.randn(num_of_timesteps, num_of_timesteps).to(device))  #[1,1]

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape  #[32,35,6,48]
        lhs = torch.matmul(torch.matmul(x.permute(0,3,2,1), self.U1), self.U2)   #
        # x:(B, N, F_in, T) -> (B, T, F_in, N)  [32,24,6,35]
        # (B, T, F_in, N)(N) -> (B,T,F_in)  [32,24,6]
        # (B,T,F_in)(F_in,N)->(B,T,N)  [32,24,35]

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)   [32,35,24]

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)  [32,24,24]

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)  [32,24,24]

        E_normalized = entmax15(E, dim=1)   #归一化

        return E_normalized


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape  # [32,8,12,24,64]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # [32,4,24,12,12]
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context

class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 用Linear来做投影矩阵
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k] [32,2,25,12,32]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, device, dropout, forward_expansion):
        super(TTransformer, self).__init__()

        # Temporal embedding One hot
        self.time_num = time_num
        #         self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding

        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, value, key, query):
        B, N, T, C = query.shape  # [32,12,24,4]
        D_T = self.temporal_embedding(torch.arange(0, T).to(self.device))  # [12,64]  # temporal embedding选用nn.Embedding
        D_T = D_T.expand(B, N, T, C)  # [32,25,12,64]

        # temporal embedding加到query。 原论文采用concatenated
        query = query + D_T  # [32,25,12,64]

        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out



class Temporal_Feature_Extraction_Module(nn.Module):
    def __init__(self,device,embed_size,num_nodes, heads, time_num, dropout, forward_expansion=4):
        super(Temporal_Feature_Extraction_Module, self).__init__()
        self.TTransformer = TTransformer(embed_size, heads, time_num, device, dropout, forward_expansion)
        self.bilinear_temporal_attention = Temporal_Attention_layer(device,embed_size,num_nodes,time_num)  # 全局时间依赖
        self.conv = nn.Conv2d(embed_size,embed_size,kernel_size=(3,1),padding=(1,0))
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        # x:[32,35,24,4]
        batch_size, num_of_vertices, num_of_timesteps, num_of_features = x.shape   #[32,12,24,512]
        x = self.TTransformer(x, x, x) + x   #[32,12,24,512]
        y = x.permute(0, 1, 3, 2)  # [32,12,512,24]

        T_attention = self.bilinear_temporal_attention(y)  # [32,12,4,24][B,T,T]
        x_TAt = torch.matmul(x.permute(0, 1, 3, 2).reshape(batch_size, -1, num_of_timesteps), T_attention).reshape(batch_size, num_of_vertices, -1,num_of_timesteps)  # [32,12,512,24]
        x_TAt = self.dropout1(self.norm1((x_TAt+y).permute(0,1,3,2))).permute(0,3,2,1) # [32,12,24,512]

        output = self.dropout2(F.relu(self.conv(x_TAt)))  # [32,4,24,12]
        return output.permute(0, 3, 2, 1) # [32,12,24,512]






