import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from visualizer import get_local
from entmax import entmax15
import numpy as np

class Gated_Fusion(nn.Module):
    def __init__(self, num_nodes, var_dim):
        super(Gated_Fusion, self).__init__()

        self.num_nodes = num_nodes
        self.var_dim = var_dim
        self.gated_fc = nn.Linear(2*var_dim, var_dim)

    def forward(self, x_adv, x_unk):
        """

        :param grad_diff: B x NDout
        :param grad_adv: B x NDout
        :return: B x NDout
        """
        B, N, D, T = x_adv.shape
        x_adv = x_adv.permute(0,1,3,2)
        x_unk = x_unk.permute(0,1,3,2)
        concat = torch.cat((x_adv, x_unk), dim=-1)  # B x N x 2
        g = torch.sigmoid(self.gated_fc(concat))

        grad_diff_adv = g * x_adv + (1 - g) * x_unk  # B x N x 1

        return grad_diff_adv.permute(0,1,3,2)  #[32,35,64,24]



class Fusion_Graph_Convolution(nn.Module):
    def __init__(self,device,K,num_nodes,in_channels, out_channels):
        super(Fusion_Graph_Convolution, self).__init__()

        self.num_nodes = num_nodes
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.alpha = nn.Parameter(torch.ones(num_nodes, num_nodes))
        self.beta = nn.Parameter(torch.ones(num_nodes, num_nodes))

    def forward(self, x, SDG, adj):


        batch_size, num_nodes, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
           x_t = x[:, :, :, time_step]  # [B, N, F_in]
           support = torch.matmul(x_t, self.weight)  # [B, N, F_out]
           gate = torch.sigmoid(self.alpha * adj + self.beta * SDG)
           spa_att_adj = gate * adj + (1 - gate) * SDG
           out = spa_att_adj.matmul(support)
           outputs.append(out.unsqueeze(-1))
        out_seq = torch.cat(outputs, dim=-1)

        return F.relu(out_seq)  # (b, N, F_out, T)  [32,35,64,24]



class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps).to(device))
        self.W2 = nn.Parameter(torch.randn(in_channels, num_of_timesteps).to(device))
        self.W3 = nn.Parameter(torch.randn(in_channels).to(device))
        self.bs = nn.Parameter(torch.randn(1, num_of_vertices, num_of_vertices).to(device))
        self.Vs = nn.Parameter(torch.randn(num_of_vertices, num_of_vertices).to(device))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = entmax15(S, dim=1)

        return S_normalized





class Spatial_Feature_Extraction_Module(nn.Module):
    def __init__(self, adj, edge_index, edge_attr,  device, in_channels,out_channels, num_nodes, num_of_timesteps, K=3,dropout=0.1, path_K=3, path_gamma=0.5, dist_lambda=0.1, use_length_weight=True):
        super(Spatial_Feature_Extraction_Module,self).__init__()
        self.bilinear_spatial_attention = Spatial_Attention_layer(device, in_channels, num_nodes, num_of_timesteps)   #全局空间依赖
        self.GCN_adv = Fusion_Graph_Convolution(device,K,num_nodes,in_channels, out_channels)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.Conv = nn.Conv2d(out_channels,out_channels,kernel_size=(3,1),padding=(1,0))

        self.adj = torch.from_numpy(adj).float().to(device)
        self.edge_index = torch.from_numpy(edge_index).to(device)
        self.edge_attr = torch.from_numpy(edge_attr).float().to(device)
        self.num_nodes = num_nodes
        self.K = K
        self.device = device
        # self.wind_std = wind_std
        # self.wind_mean = wind_mean

    def creat_support_matrix(self, last_wind_vars):
        # [32,35,24,2]
        batch_size = last_wind_vars.shape[0]
        edge_src, edge_target = self.edge_index  # 图的边索引
        num_edges = edge_src.shape[0]

        # 计算正向影响（src--> target）
        node_src = last_wind_vars[:, edge_src, :]  # 起点
        node_target = last_wind_vars[:, edge_target, :]  # 终点
        
        # 边的方向 (来自 edge_attr)
        edge_dir_deg = self.edge_attr[:, 2].unsqueeze(0).repeat(batch_size, 1)  # [B, E]
        # 边的长度/距离 (来自 edge_attr)
        edge_length = self.edge_attr[:, 1].unsqueeze(0).repeat(batch_size, 1)  # [B, E]

        # 2. 将风向转换为u/v分量（气象风向是来向，需转换为矢量去向）
        def wind_to_uv(wind_dir, wind_speed):
            rad = torch.deg2rad(wind_dir)
            u = wind_speed * torch.sin(rad)  # 东向分量（负号因风向是来向）
            v = wind_speed * torch.cos(rad)  # 北向分量
            return u, v

        u_src, v_src = wind_to_uv(node_src[:, :, 0], node_src[:, :, 1])
        u_tgt, v_tgt = wind_to_uv(node_target[:, :, 0], node_target[:, :, 1])

        # 3. 计算平均u/v分量（综合风速矢量）
        u_avg = (u_src + u_tgt) / 2
        v_avg = (v_src + v_tgt) / 2
        wind_speed_avg = torch.sqrt(u_avg ** 2 + v_avg ** 2)  # 综合风速大小

        # 5. 计算综合风向角（气象学定义，正北为0°）
        wind_dir_avg = torch.rad2deg(torch.atan2(u_avg, v_avg)) % 360  # [batch, num_edges]

        # 6. 计算风速在边方向上的投影强度
        angle_diff = (wind_dir_avg - edge_dir_deg + 180) % 360 - 180   # 风向与边方向的夹角
        projection = wind_speed_avg * torch.cos(torch.deg2rad(angle_diff)) # [batch, num_edges]
        # 边段积分近似：顺风方向的投影乘以边长，并可选指数衰减
        edge_weight_seg = torch.clamp(projection, min=0.0)


        # Step 5. 构造邻接矩阵 (batch版)
        adj = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=self.device)
        adj[torch.arange(batch_size).unsqueeze(1), edge_src, edge_target] = edge_weight_seg
        
        # Step 6. 加自环
        I = torch.eye(self.num_nodes, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + I
        
        # Step 7. 对称归一化  D^{-1/2} A D^{-1/2}
        deg = torch.sum(adj, dim=-1)  # [B, N]
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # [B, N, N]
        adj_norm = torch.bmm(torch.bmm(D_inv_sqrt, adj), D_inv_sqrt)


        self._supports_adv = adj_norm  # [B, N, N]


    def forward(self,x):
        # x:[B,N,F,T]
        x =x.permute(0,1,3,2)  #[]
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # spatio-temporal dependence weight
        SDG = self.bilinear_spatial_attention(x)  # [B,N,N]
        x_s = self.GCN_adv(x, SDG, self._supports_adv)  #[32,35,64,24]

        # residual_layer
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # [32,512,21,24]
        x_residual = self.dropout(self.ln(F.relu(x_residual + x_s.permute(0,2,1,3)).permute(0,2,3,1)).permute(0,3,1,2))
        output = self.dropout(F.relu(self.Conv(x_residual))) #[32,512,21,24]
        return output.permute(0,2,3,1)  #[32,512,21,24]-->[32,21,24,512]
