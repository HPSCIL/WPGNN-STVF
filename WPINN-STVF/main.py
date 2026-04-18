import copy
import time
import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from Data_Process import pro_data,MyDataset
from torch.utils.data import random_split
from utils import get_logger,All_Metrics,init_seed
from model import MODEL
from get_adjacency_matrix import get_adjacency_matrix
import matplotlib.pyplot as plt

Model = 'WG-STVFNet'
Dataset = 'Beijing'
device = torch.device('cuda')
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--d_model', type=int, default=512)
args.add_argument('--d_k', type=int, default=64)
args.add_argument('--n_heads', type=int, default=8)
args.add_argument('--var_dim', type=int, default=4)
args.add_argument('--hid_dim', type=int, default=512)
args.add_argument('--num_nodes', type=int, default=35)
args.add_argument('--lag', type=int, default=24)
args.add_argument('--pre_len', type=int, default=24)
args.add_argument('--d_ff', type=int, default=512)
args.add_argument('--device', type=str, default=device)
args.add_argument('--learning_rate', type=float, default=0.001)
args.add_argument('--debug', type=eval, default=False)
args.add_argument('--log_dir', type=str, default='./')
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=5, type=int)
args.add_argument('--epochs', default=200, type=int)
args.add_argument('--cheb_k', default=3, type=int)
args.add_argument('--seed', default=100, type=int)
args.add_argument('--batch_size', default=32, type=int)
args.add_argument('--dropout', default=0.3, type=int)
args.add_argument('--weight_decay', default=0.001, type=int)
args = args.parse_args()

init_seed(args.seed)


# Station_info = np.load('D:\周欣萌的文件\第三篇论文/72小时实验\数据/wh_72_data.npy')

# dataframe = pd.read_excel('D:\周欣萌的文件\实验\实验数据\伦敦/伦敦多污染物数据.xlsx',sheet_name="源数据")  #[12,-1,12]
# Station_info = dataframe[['PM2.5', 'PM10', 'O3', 'NO2']]
# dataset = Station_info.values
# dataset = dataset.astype('float32')
# dataset = np.reshape(dataset, (22, -1, 4))
# Station_info = dataset.transpose(1,0,2)  #[-1,21,4 ]

dataframe = pd.read_excel('D:\周欣萌的文件\第三篇论文\实验数据\实验数据汇总\北京/北京数据.xlsx')  # [12,-1,12]
data = dataframe[['PM2.5', 'PM10', 'O3', 'NO2','Wind_dir','Wind_speed']]
dataset = data.values
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (35, -1, 6))
data = dataset.transpose(1,0,2)  #[-1,21,4 ]


#
# dataframe = pd.read_excel('D:\周欣萌的文件\第三篇论文\实验数据\实验数据汇总\武汉/武汉数据.xlsx')  # [12,-1,12]
# Station_info = dataframe[['PM2.5', 'PM10', 'O3', 'NO2','Wind_dir','Wind_speed']]#,'Humidity','Pressure','Temperature','Wind_dir','Wind_speed']]
# dataset = Station_info.values
# dataset = dataset.astype('float32')
# dataset = np.reshape(dataset, (21, -1, 6))
# Station_info = dataset.transpose(1, 0, 2)  # [-1,21,4]
# np.save('Station_info/data_wh.npy', Station_info)
# Station_info = np.load('Station_info/data_ld.npy')


# 构建要用到的邻接矩阵
adj_mx, edge_index, edge_attr, node_pos = get_adjacency_matrix()
args.adj_mx = adj_mx
args.edge_index = edge_index
args.edge_attr = edge_attr
args.node_pos = node_pos


X,Y= pro_data(data,args)
#
# 创建Dataset对象
dataset = MyDataset(X, Y)

# Setting the size of the training set, test set and validation set
train_ratio = 0.6
val_ration = 0.2
test_ration = 0.2
train_size = int(len(dataset) * train_ratio)
val_size = int(len(dataset) * val_ration)
test_size = len(dataset) - train_size - val_size

train_start = 0
train_end = train_size
val_end = train_size + val_size
test_end = len(dataset)  

train_dataset = Subset(dataset, range(train_start, train_end))
val_dataset = Subset(dataset, range(train_end, val_end))
test_dataset = Subset(dataset, range(val_end, test_end))

#
# 现在你可以使用train_dataset和test_dataset来创建DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)




net = MODEL(args).to(device)

for p in net.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

# Define loss function and Optimization Function
criterion = torch.nn.L1Loss().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)



# Define store file path
result_filename = Dataset+'_in_'+str(args.lag)+'h_out_'+str(args.pre_len)+'h'
print(result_filename)
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join('./results', result_filename)
print(log_dir)
if os.path.exists(log_dir) == False:
    os.makedirs(log_dir)
logger = get_logger(log_dir, name=Model, debug=args.debug)
logger.info('Experiment log path in: {}'.format(args.log_dir))


# Define best model path
best_path = os.path.join(log_dir, 'best_model.pth')
print(best_path)


# Train and val
best_loss = np.inf
start_time = time.time()
train_losses = []
val_losses = []
for epoch in range(1,args.epochs+1):
    total_loss = 0
    test_outputs = torch.empty([0, args.num_nodes,4, args.pre_len]).to(device)
    test_target = torch.empty([0, args.num_nodes,4, args.pre_len]).to(device)
    T_Att = []
    train_per_epoch = len(train_loader)
    # Train

    for batch_index, batch_data in enumerate(train_loader):
        X, Y = batch_data
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        output= net(X)  # [32,35,12]
        loss = criterion(output, Y)
        # # 添加L2正则化项
        # loss = loss + l2_regularization_term
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_index % args.log_step == 0:
            logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_index, train_per_epoch, loss.item()))
    train_epoch_loss = total_loss / train_per_epoch
    train_losses.append(train_epoch_loss)

    logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

    # val
    val_loader = val_loader
    net.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            X, Y = batch_data
            X = X.to(device)
            Y = Y.to(device)
            output = net(X)
            loss_val = criterion(output, Y)
            # loss_val = MSE_torch(output, Y)
            total_val_loss += loss_val.item()
    val_epoch_loss = total_val_loss / len(val_loader)
    val_losses.append(val_epoch_loss)
    logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch,val_epoch_loss))
    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        not_improved_count = 0
        best_state = True
    else:
        not_improved_count += 1
        best_state = False
    if args.early_stop:
        if not_improved_count == args.early_stop_patience:
            print("Validation performance didn\'t improve for {} epochs. "
                  "Training stops.".format(args.early_stop_patience))
            break
    if best_state == True:
        print('*********************************Current best model saved!')
        best_model = copy.deepcopy(net.state_dict())
torch.save(best_model, best_path)

logger.info("Saving current best model to " + best_path)

plt.figure(figsize=(8,6))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
plt.show()

# Test
net.load_state_dict(torch.load(best_path))
net.eval()
with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_loader):
        test_X, test_Y= batch_data
        test_X = test_X.to(device)
        test_Y = test_Y.to(device)
        test_output = net(test_X)

        test_outputs = torch.cat((test_outputs, test_output), 0)
        test_target = torch.cat((test_target, test_Y), 0)

    mae1, rmse1, mape1, r21 = All_Metrics(test_outputs[:, :, 0, :], test_target[:, :, 0, :])
    mae2, rmse2, mape2, r22 = All_Metrics(test_outputs[:, :, 1, :], test_target[:, :, 1, :])
    mae3, rmse3, mape3, r23 = All_Metrics(test_outputs[:, :, 2, :], test_target[:, :, 2, :])
    mae4, rmse4, mape4, r24 = All_Metrics(test_outputs[:, :, 3, :], test_target[:, :, 3, :])

    logger.info('loss_PM25:{},rmse_PM25:{},mape_PM25:{},r2_PM25:{}'.format(mae1, rmse1, mape1, r21))
    logger.info('loss_PM10:{},rmse_PM10:{},mape_PM10:{},r2_PM10:{}'.format(mae2, rmse2, mape2, r22))
    logger.info('loss_O3:{},rmse_O3:{},mape_O3:{},r2_O3:{}'.format(mae3, rmse3, mape3, r23))
    logger.info('loss_NO2:{},rmse_NO2:{},mape_NO2:{},r2_NO2:{}'.format(mae4, rmse4, mape4, r24))
