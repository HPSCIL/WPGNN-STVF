import torch.nn as nn
from Layers import Decouple_Layer
from CrossAttention import CrossAttention


class Decouple_Spatial_Temporal_Variables(nn.Module):
    def __init__(self,adj, edge_index, edge_attr, var_dim,hid_dim,num_nodes,lag,d_k,d_model,n_heads,device, dropout):
        super(Decouple_Spatial_Temporal_Variables,self).__init__()
        self.decouple_layer = Decouple_Layer(adj, edge_index, edge_attr, var_dim,hid_dim,num_nodes,lag,d_k,d_model,n_heads,device, dropout)

    def forward(self,x):
        # [32,35,6,24]
        x = x.permute(0,1,3,2)  #[32,35,24,6]
        temporal_att, spatial_att, variable_att = self.decouple_layer(x)
        return temporal_att, spatial_att, variable_att





class Multidimensional_Feature_Fusion(nn.Module):
    def __init__(self, n_head, num_nods,d_model, d_k, dropout):
        super(Multidimensional_Feature_Fusion,self).__init__()
        self.Crossatt_layer1 = CrossAttention(n_head, d_model, d_k, dropout)

        self.prediction = nn.Conv2d(num_nods,num_nods,kernel_size=(1,d_model-3))
        # 在 __init__ 中定义 prediction 层
        # self.prediction = nn.Sequential(
        #     nn.Linear(24 * 512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 4 * 24)
        # )



    def forward(self, x1,x2,x3):
        attention12 = self.Crossatt_layer1(x2,x1,x1)  # [32,35,24,512]
        attention23 = self.Crossatt_layer1(x3,attention12,attention12) #[32,35,24,512]

        output = self.prediction(attention23)  #[32,35,4,48]

        # B, N, T, D = attention23.shape  # [32, 35, 24, 512]
        # x = attention23.reshape(B, N, T * D)  # -> [32, 35, 12288]
        # x = self.prediction(x)  # -> [32, 35, 192]  （4 * 48）
        # output = x.view(B, N, 4, 24)  # -> [32, 35, 4, 48]
        #output = self.prediction2(output.permute(0,1,3,2))   #[32,512,12,35]
        # return output.permute(0,3,1,2) #.transpose(1,3)
        # output = self.prediction1(output)  #[32,35,24,4)
        # output = self.prediction2(output.permute(0,1,3,2))
        return output.permute(0,1,3,2)




class MODEL(nn.Module):
    def __init__(self,args):
        super(MODEL, self).__init__()
        self.var_dim = args.var_dim
        self.hid_dim = args.hid_dim
        self.num_nodes =args.num_nodes
        self.lag =args.lag
        self.d_k =args.d_k
        self.d_model =args.d_model
        self.d_ff =args.d_ff
        self.n_heads =args.n_heads
        self.device = args.device
        self.pre_len = args.pre_len
        self.edge_index = args.edge_index
        self.edge_attr = args.edge_attr
        self.adj = args.adj_mx
        self.dropout = args.dropout

        self.Multidimensional_Feature_Decouple = Decouple_Spatial_Temporal_Variables(
                                                   self.adj, self.edge_index,
                                                   self.edge_attr, self.var_dim,
                                                   self.hid_dim, self.num_nodes, self.lag,
                                                   self.d_k, self.d_model, self.n_heads,
                                                   self.device, self.dropout)

        self.Multidimensional_Feature_Fusion = Multidimensional_Feature_Fusion(
                                                   self.n_heads,
                                                   self.num_nodes, self.d_model,
                                                   self.d_k, self.dropout)

    def forward(self,x):
      # x: [32,35,6,24]
      Temporal, Spatial, Variable = self.Multidimensional_Feature_Decouple(x)
      output = self.Multidimensional_Feature_Fusion(Temporal, Spatial, Variable)

      return output

