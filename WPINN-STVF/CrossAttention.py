import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from visualizer import get_local

class CrossAttention(nn.Module):
    def __init__(self, n_head,embed_size, d_k, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(embed_size, n_head * d_k)
        self.w_ks = nn.Linear(embed_size, n_head * d_k)
        self.w_vs = nn.Linear(embed_size, n_head * d_k)

        self.fc = nn.Linear(n_head * d_k, embed_size)
        #
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        # self.scale = nn.Parameter(torch.ones(1) / math.sqrt(d_k))

    def forward(self, q,k,v):
        #[32, 35, 24, 4]
        d_k, n_head = self.d_k, self.n_head



        B,N,T,D = q.size()
        residual = q

        q = self.w_qs(q).view(B,N, T, n_head, d_k)  #[32,24,8,64]
        k = self.w_ks(k).view(B,N, T, n_head, d_k)  #[32,24,8,64]
        v = self.w_vs(v).view(B,N, T, n_head, d_k)  #[32,24,8,64]

        q = q.permute(0, 3,1,2,4)  #[32,8,35,24,64]
        k = k.permute(0, 3,1,2,4)
        v = v.permute(0, 3,1,2,4)

        attn = torch.matmul(q, k.transpose(-2, -1))  #[32,35,24,24]
        attn = attn / math.sqrt(d_k)
        scores = self.dropout(F.softmax(attn, dim = -1))
        output = torch.matmul(scores, v)  #[256,8,12,24,64]
        # output = output+residual

        output = output.permute(0,2,3,1,4).contiguous().view(B, N,T, n_head * d_k) #[32,12,24,512]
        output = self.fc(output)  #[32,12,24,512]
        output = self.dropout(F.relu(output))
        output = self.layer_norm(output + residual)
        output = output.view(B,T,N,D).permute(0,2,1,3)
        return output