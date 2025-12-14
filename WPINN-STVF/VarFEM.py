import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from visualizer import get_local

class Variable_Feature_Extraction_Module(nn.Module):
    def __init__(self, n_head, num_nods, lag, d_k, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(num_nods*lag, n_head * d_k)
        self.w_ks = nn.Linear(num_nods*lag, n_head * d_k)
        self.w_vs = nn.Linear(num_nods*lag, n_head * d_k)

        self.fc = nn.Linear(n_head * d_k, num_nods*lag)

        self.layer_norm = nn.LayerNorm(num_nods*lag)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        d_k,  n_head = self.d_k, self.n_head

        B,N,T,D = x.size()
        x = x.permute(0,3,1,2).reshape(B,D,-1)   #将变量用T*D的向量表示  [32,4,504]


        residual = x
        q = self.w_qs(x).view(B, D, n_head, d_k)  #[32,4,8,64]  对变量向量特征提取
        k = self.w_ks(x).view(B, D, n_head, d_k)  #[32,19,8,64]
        v = self.w_vs(x).view(B, D, n_head, d_k)  #[32,19,8,64]

        q = q.permute(0, 2, 1, 3)  #[32,8,4,64]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1))  #[32,8,4,4]
        attn = attn / math.sqrt(d_k)
        scores = self.dropout1(F.softmax(attn, dim = -1))
        output = torch.matmul(scores, v)  #[256,8,4,64]

        output = output.transpose(1, 2).contiguous().view(B, -1, n_head * d_k) #[256,4,512]
        output = self.fc(output)  #[32,4,35*24]
        output = self.dropout2(F.relu(output))
        output = self.layer_norm(output + residual)
        output = output.view(B, D, N, T).permute(0,2,3,1)

        return output #[32,35,24,4]