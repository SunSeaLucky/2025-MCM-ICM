from statics.Statics import Statics

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  
import seaborn as sns  

# 定义 LSTM 模型  
class LSTM(nn.Module):  
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):  
        super(LSTM, self).__init__()  
        self.hidden_dim = hidden_dim  
        self.num_layers = num_layers  
        
        # LSTM 层  
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)  
        
        # 全连接层  
        self.fc = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):  
        # 初始化隐藏状态和细胞状态  
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  
        
        # LSTM 前向传播  
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # 输出和最后的隐藏状态  
        
        # 仅使用最后一个时间步的输出  
        # out = self.fc(out[:, -1, :])  # (batch_size, output_dim)  
        out = self.fc(out[:, :, :])  # (batch_size, output_dim)  
        return out

class LSTMAdvanced:
    def __init__(self):
        self.scaler = MinMaxScaler()
        # 嵌入向量的维度
        self.embedding_dim = 16
        # 目标特征
        self.output_feature = ['TotalMedal', 'GoldMedal']
        # 国家数量
        self.batch_size = 233
        # 时间序列长度
        self.sequence_length = 30
        # 数值型特征列
        self.input_numeric_features = ['strong_point', 'hhi', 'award_rate', 'participate_num', 'history_performance']
        self.output_numeric_features = [self.output_feature[0]]
        # 数据集
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.batch_size = 233  # 国家数量  
        self.sequence_length = 30 # 时间序列长度  
        self.input_dim = 37  # 输入特征维度  

        # 模型参数  
        self.hidden_dim = 256  # LSTM 隐藏层维度  
        self.num_layers = 1  # LSTM 层数  
        self.output_dim = 1  # 输出维度（假设是回归任务）  
        self.num_epochs = 100  # 训练轮数  
        self.learning_rate = 0.01  # 学习率
        
        # 定义模型
        self.model = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, output_dim=self.output_dim)

        # 损失记录
        self.hist = None
        
    def train(self):
        assert self.x_train is not None and self.y_train is not None, "Training set is empty"
        assert self.x_test is not None and self.y_test is not None, "Test set is empty"

        # 初始化模型、损失函数和优化器  
        model = self.model
        criterion = torch.nn.MSELoss()  # 均方误差损失  
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)  # Adam 优化器  

        # 训练模型  
        hist = np.zeros(self.num_epochs)  # 用于记录每个 epoch 的损失  
        start_time = time.time()  

        for t in range(self.num_epochs):  
            # 前向传播  
            y_train_pred = model(self.x_train)  
            
            # 计算损失  
            loss = criterion(y_train_pred, self.y_train)  
            print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()  

            # 反向传播  
            optimiser.zero_grad()  # 梯度清零  
            loss.backward()  # 计算梯度  
            optimiser.step()  # 更新参数  
            
        self.hist = hist
        
        # 打印训练时间
        training_time = time.time() - start_time  
        print("Training time: {} seconds".format(training_time))

    def dataset_construct(self):
        sta = Statics()        
        data = sta.raw_dataset.csv_file        
        all_countries = sta.get_all_countries()
        country_to_index = {country: idx for idx, country in enumerate(all_countries)}
        data['NOC_index'] = data['NOC'].map(country_to_index)
        data['host_index'] = data['host'].map(country_to_index)

        # 假设国家集合大小
        num_countries = len(all_countries)

        # 定义嵌入层
        noc_embedding = nn.Embedding(num_countries, self.embedding_dim)  
        host_embedding = nn.Embedding(num_countries, self.embedding_dim)

        # 对数值型特征进行归一化
        data[self.input_numeric_features] = self.scaler.fit_transform(data[self.input_numeric_features])
        data[self.output_numeric_features] = self.scaler.fit_transform(data[self.output_numeric_features])

        # 构造 NOC 和 host 的索引序列
        noc_indices = torch.tensor(data['NOC_index'].values).view(self.batch_size, self.sequence_length)
        host_indices = torch.tensor(data['host_index'].values).view(self.batch_size, self.sequence_length)
        numeric_inputs = torch.tensor(data[self.input_numeric_features].values).view(self.batch_size, self.sequence_length, len(self.input_numeric_features))
        numeric_outputs = torch.tensor(data[self.output_numeric_features].values).view(self.batch_size, self.sequence_length, 1)

        # 获取嵌入向量
        noc_embedded = noc_embedding(noc_indices)  # (batch_size, sequence_length, embedding_dim)  
        host_embedded = host_embedding(host_indices)  # (batch_size, sequence_length, embedding_dim)  

        # 拼接嵌入向量和数值特征
        combined_inputs = torch.cat((noc_embedded, host_embedded, numeric_inputs), dim=2)  # (batch_size, sequence_length, total_dim)  
         
        x = combined_inputs
        y = numeric_outputs

        # 按时间步划分  
        train_ratio = 0.7  # 70% 的时间步用于训练  
        train_length = int(self.sequence_length * train_ratio)  # 训练集时间步数  
        test_length = self.sequence_length - train_length  # 测试集时间步数  

        # 划分训练集和测试集  
        x_train = x[:, :train_length, :]  # (batch_size, train_length, input_dim)  
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = y[:, :train_length, :]  # (batch_size, train_length, 1)  
        y_train = torch.tensor(y_train, dtype=torch.float32)

        x_test = x[:, train_length:, :]  # (batch_size, test_length, input_dim)  
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = y[:, train_length:, :]  # (batch_size, test_length, 1)  
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def draw_cmp_plot(self, y_train_pred):
        # 假设 y_train_pred 和 y_train 的形状为 (233, 21, 1)  
        # 转换为 NumPy 数组  
        y_train_pred_np = y_train_pred.detach().numpy().squeeze()  # (233, 21)  
        y_train_np = self.y_train.detach().numpy().squeeze()  # (233, 21)  

        # 选择一个样本（例如第一个国家）进行可视化  
        sample_index = 20
        y_train_pred_sample = y_train_pred_np[sample_index]  # (21,)  
        y_train_sample = y_train_np[sample_index]  # (21,)  

        # 如果需要反归一化，可以使用 scaler.inverse_transform  
        # 假设 scaler 是用于归一化的 MinMaxScaler  
        y_train_pred_sample = self.scaler.inverse_transform(y_train_pred_sample.reshape(-1, 1)).squeeze()  
        y_train_sample = self.scaler.inverse_transform(y_train_sample.reshape(-1, 1)).squeeze()  

        # 设置绘图风格  
        sns.set_style("darkgrid")  

        # 创建图形  
        fig = plt.figure()  
        fig.subplots_adjust(hspace=0.2, wspace=0.2)  

        # 绘制预测值和真实值的对比图  
        plt.subplot(1, 2, 1)  
        ax = sns.lineplot(x=range(len(y_train_sample)), y=y_train_sample, label="True Values", color='royalblue')  
        ax = sns.lineplot(x=range(len(y_train_pred_sample)), y=y_train_pred_sample, label="Predictions (LSTM)", color='tomato')  
        ax.set_title('Prediction vs True Values', size=14, fontweight='bold')  
        ax.set_xlabel("Time Steps", size=14)  
        ax.set_ylabel("Value", size=14)  

        # 绘制训练损失  
        plt.subplot(1, 2, 2)  
        ax = sns.lineplot(data=self.hist, color='royalblue')  
        ax.set_xlabel("Epoch", size=14)  
        ax.set_ylabel("Loss", size=14)  
        ax.set_title("Training Loss", size=14, fontweight='bold')  

        # 设置图形大小  
        fig.set_figheight(6)  
        fig.set_figwidth(16)  

        # 显示图形  
        plt.show()