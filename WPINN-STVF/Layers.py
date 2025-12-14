import torch.nn as nn
import torch
from TemFEM import Temporal_Feature_Extraction_Module
from SpaFEM import Spatial_Feature_Extraction_Module
from VarFEM import Variable_Feature_Extraction_Module
from Embedding import DataEmbedding

class Decouple_Layer(nn.Module):
    def __init__(self,adj, edge_index, edge_attr, var_dim,hid_dim,num_nodes,lag,d_k,d_model,n_heads,device, dropout):
        super(Decouple_Layer,self).__init__()

        self.var_dim = var_dim
        self.data_embedding = DataEmbedding(var_dim,d_model)

        self.temporal_layer = Temporal_Feature_Extraction_Module(device,embed_size=d_model,num_nodes=num_nodes, heads=n_heads, time_num=24, dropout=dropout, forward_expansion=4)

        self.spatial_layer = Spatial_Feature_Extraction_Module(adj, edge_index, edge_attr, device,var_dim,hid_dim,num_nodes,lag)

        self.variant_layer = Variable_Feature_Extraction_Module(n_heads, num_nodes, lag, d_k)
        self.variant_conv = nn.Linear(var_dim,d_model)

    def aggregate_wind(self, wind_vars):
        wind_dir = wind_vars[:,:,:,0]
        wind_speed = wind_vars[:,:,:,1]

        # 将风向转换为单位向量（气象风向是来向，需转换为去向）
        rad = torch.deg2rad(wind_dir)
        u = -torch.sin(rad)   # 东向分量
        v = -torch.cos(rad)   # 北向分量
        u_speed = wind_speed * u
        v_speed = wind_speed * v

        # Step 3: 定义时间权重（线性增长：后期风更重要）
        weights = torch.linspace(1, 2, steps=24)  # [24]
        weights = weights / weights.sum()  # 归一化，确保和为1
        weights = weights.to(u.device)  # 放到相同设备上

        # 计算时间维度上的综合向量分量
        aggregated_u = (u_speed * weights).sum(dim=2)
        aggregated_v = (v_speed * weights).sum(dim=2)
        # 计算时间维度上的综合向量分量
        # aggregated_u = u_speed.sum(dim=2)
        # aggregated_v = v_speed.sum(dim=2)

        # 将综合向量转换回风向角度和风速
        weighted_speed = torch.sqrt(aggregated_u ** 2 + aggregated_v ** 2)
        weighted_dir = (torch.rad2deg(torch.atan2(aggregated_u, aggregated_v)) + 360) % 360
        return torch.stack([weighted_dir, weighted_speed], dim=-1)

    def forward(self,x):

        pollutants = x[:, :, :, :self.var_dim]
        # wind_vars = x[:, :, -1, -2:]  #[32,35,2]
        # wind_vars = torch.mean(x, dim=2)
        # wind_vars_first = x[:, :, 0, -2:]  # [32,35,2]
        # wind_vars = (wind_vars_first + wind_vars_last) / 2
        wind_vars = x[:, :, :, -2:]  # [32,35,2]
        wind_vars = self.aggregate_wind(wind_vars)


        x1 = pollutants
        x2 = pollutants
        x3 = pollutants
        x1 = self.data_embedding(x1)  #[32,12,24,512]
        x1 = self.temporal_layer(x1)
        self.spatial_layer.creat_support_matrix(wind_vars)
        x2 = self.spatial_layer(x2)  #[32,12,24,512]
        x3 = self.variant_layer(x3)  #[32,21,24,4]
        x3 = self.variant_conv(x3)  #[32,12,24,512]

        return x1, x2, x3



