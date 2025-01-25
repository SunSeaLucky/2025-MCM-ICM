from statics.Statics import Statics
from preprocess.Preprocessor import InputFuture
from typing import Literal
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  
import seaborn as sns  

import plotly.graph_objects as go  
import numpy as np  

# 定义 LSTM 模型  
class LSTM(nn.Module):  
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):  
        super(LSTM, self).__init__()  
        self.hidden_dim = hidden_dim  
        self.num_layers = num_layers  
        
        # LSTM 层  
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)  
        
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
        self.scaler = StandardScaler()
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
        self.x_input_future = None
        self.batch_size = 233  # 国家数量  
        self.sequence_length = 30 # 时间序列长度  
        self.input_dim = 37  # 输入特征维度  

        # 模型参数  
        self.hidden_dim = 256  # LSTM 隐藏层维度  
        self.num_layers = 1  # LSTM 层数  
        self.output_dim = 1  # 输出维度（假设是回归任务）  
        self.num_epochs = 100  # 训练轮数  
        self.learning_rate = 0.01  # 学习率
        self.dropout = 0.4  # Dropout 率
        
        # 定义模型
        self.model = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, output_dim=self.output_dim, dropout=self.dropout)

        # Loss 记录
        self.hist = None
        
    def monte_carlo_sampling(self, model, x, sample_index: int = 20, num_samples: int = 100):  
        """  
        使用 MC Dropout 对模型进行多次采样，生成预测分布。  
        
        参数：  
        - model: 带 Dropout 的模型  
        - x: 输入数据 (batch_size, sequence_length, input_dim)  
        - num_samples: 采样次数  
        
        返回：  
        - mean: 预测均值 (batch_size, output_dim)  
        - std: 预测标准差 (batch_size, output_dim)  
        """        
        # 进行多次采样  
        predictions = []  
        for _ in range(num_samples):  
            with torch.no_grad():  # 禁用梯度计算以节省内存  
                # y_test_pred_sample = model(x).detach().numpy().squeeze()[sample_index]  # (21,)  
                # y_test_pred_sample = self.scaler.inverse_transform(y_test_pred_sample.reshape(-1, 1)).squeeze()                
                predictions.append(self.transform_from_tensor_data(model(x), sample_index))
        
        # 计算均值和标准差
        return np.mean(predictions), np.std(predictions)
        
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
        
    def __draw_cmp_plot__(self, y_train_pred):
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
    
    def transform_from_tensor_data(self, tensor_data, sample_index: int = 20):
        '''
        把模型输出的 tensor 数据转换为更易懂的数据形式（同时反归一化）。
        
        实际上，即给出指定国家的预测结果。
        '''
        tensor_data = tensor_data.detach().numpy().squeeze()[sample_index].reshape(-1, 1)
        tensor_data = self.scaler.inverse_transform(tensor_data).squeeze()
        return tensor_data        
    
    def draw_cmp(self, sample_index: int = 20):
        '''
        绘制预测值和真实值的对比图。
        '''
       
        # y_train_pred_np = y_train_pred.detach().numpy().squeeze()  # (233, 21)  
        # y_train_np = self.y_train.detach().numpy().squeeze()  # (233, 21) 
        # y_train_pred_sample = y_train_pred_np[sample_index]  # (21,)  
        # y_train_sample = y_train_np[sample_index]  # (21,)  

        # If inverse normalization is required  
        # y_train_pred_sample = self.scaler.inverse_transform(y_train_pred_sample.reshape(-1, 1)).squeeze()  
        # y_train_sample = self.scaler.inverse_transform(y_train_sample.reshape(-1, 1)).squeeze()
        
        
        
        y_train_pred_sample = np.concatenate((self.transform_from_tensor_data(self.model(self.x_train), sample_index),
                                             self.transform_from_tensor_data(self.model(self.x_test), sample_index)))
        y_train_sample = np.concatenate((self.transform_from_tensor_data(self.y_train, sample_index),
                                        self.transform_from_tensor_data(self.y_test, sample_index)))

        # Define a professional color palette  
        colors = {  
            "true_values": "#1f77b4",  # Coolors Blue  
            "predictions": "#ff7f0e",  # Coolors Orange  
            "loss": "#2ca02c"          # Coolors Green  
        }  

        # Create the Prediction vs True Values plot  
        fig1 = go.Figure()  

        fig1.add_trace(go.Scatter(  
            x=np.arange(len(y_train_sample)),  
            y=y_train_sample,  
            mode='lines+markers',  
            name='True Values',  
            line=dict(color=colors["true_values"], width=3),  
            marker=dict(symbol='circle', size=8)  
        ))  

        fig1.add_trace(go.Scatter(  
            x=np.arange(len(y_train_pred_sample)),  
            y=y_train_pred_sample,  
            mode='lines+markers',  
            name='Predictions (LSTM)',  
            line=dict(color=colors["predictions"], width=3, dash='dot'),  
            marker=dict(symbol='square', size=8)  
        ))
        
        # 添加竖线  
        fig1.add_shape(  
            type="line",  
            x0=21, x1=21,  # x 坐标范围  
            y0=0, y1=max(max(y_train_sample), max(y_train_pred_sample)),  # y 坐标范围  
            line=dict(color="purple", width=3, dash="dash"),  # 紫色虚线  
            name="Vertical Line"  
        ) 
        
        # 添加文字标注  
        fig1.add_annotation(  
            x=21,  # x 坐标  
            y=max(max(y_train_sample), max(y_train_pred_sample)) + 1,  # y 坐标，稍微高于竖线  
            text="Train-Test Split",  # 标注文字  
            showarrow=False,  # 不显示箭头  
            font=dict(color="purple", size=12),  # 设置字体颜色和大小  
            align="center"  # 居中对齐  
        ) 

        # Update layout for Prediction vs True Values  
        fig1.update_layout(  
            title=dict(  
                text='Prediction vs True Values',  
                font=dict(size=20, family='Times New Roman'),  
                x=0.5  # Center title  
            ),  
            xaxis_title='Time Steps',  
            yaxis_title='Value',  
            font=dict(family='Times New Roman', size=14),  
            legend=dict(  
                title="Legend",  
                font=dict(size=12),  
                bordercolor="Black",  
                borderwidth=1,  
                x=0.02,  # Position legend to avoid overlap  
                y=0.98  
            ),  
            width=800,  
            height=500,  
            margin=dict(l=50, r=50, t=50, b=50),  
            plot_bgcolor='white',  
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),  
            yaxis=dict(showgrid=True, gridcolor='lightgrey')  
        )  

        fig1.show()  
        
    def draw_loss(self):
        '''
        绘制训练损失图。
        '''
        colors = {  
            "true_values": "#1f77b4",  # Coolors Blue  
            "predictions": "#ff7f0e",  # Coolors Orange  
            "loss": "#2ca02c"          # Coolors Green  
        }
        fig2 = go.Figure()  

        fig2.add_trace(go.Scatter(  
            x=np.arange(len(self.hist)),  
            y=self.hist,  
            mode='lines+markers',  
            name='Loss',  
            line=dict(color=colors["loss"], width=3),  
            marker=dict(symbol='triangle-up', size=8)  
        ))  
        fig2.update_layout(  
            title=dict(  
                text='Training Loss',  
                font=dict(size=20, family='Times New Roman'),  
                x=0.5  # Center title  
            ),  
            xaxis_title='Epoch',  
            yaxis_title='Loss',  
            font=dict(family='Times New Roman', size=14),  
            legend=dict(  
                title="Legend",  
                font=dict(size=12),  
                bordercolor="Black",  
                borderwidth=1,  
                x=0.02,  
                y=0.98  
            ),  
            width=800,  
            height=500,  
            margin=dict(l=50, r=50, t=50, b=50),  
            plot_bgcolor='white',  
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),  
            yaxis=dict(showgrid=True, gridcolor='lightgrey')  
        )  
        
        fig2.show()

    def get_medal_board(self, output, year: int = 2024, type = Literal['train', 'test', 'pred']):
        sta = Statics()
        
        valid_years = sta.get_valid_years()
        all_countries = sta.get_all_countries()
        
        if type == 'train':
            base = 0
        elif type == 'test':
            base = int(len(valid_years) * 0.7)
        else:
            valid_years.append(2028)
            base = len(valid_years)
        
        idx = valid_years.index(year)
        
        arr = []
        for i in range(self.batch_size):
            country = all_countries[i]
            medal = self.transform_from_tensor_data(output, i)
            if medal.size > 1:
                medal = medal[idx - base]
            if type == 'pred':
                medal *= 0.15
            # print(idx - base)
            arr.append([country, medal])
            
        pd.DataFrame(arr, columns=['Country', 'Medal'])\
          .sort_values(by='Medal', ascending=False)\
          .to_csv('./mid_data/medal_board_%d.csv' % year, index=False)
        
    def input_future_construct(self, year: int =2028):
        sta = Statics()
        data = InputFuture().csv_file
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
        noc_indices = torch.tensor(data['NOC_index'].values).view(self.batch_size, 1)
        host_indices = torch.tensor(data['host_index'].values).view(self.batch_size, 1)
        numeric_inputs = torch.tensor(data[self.input_numeric_features].values).view(self.batch_size, 1, len(self.input_numeric_features))
        # numeric_outputs = torch.tensor(data[self.output_numeric_features].values).view(self.batch_size, self.sequence_length, 1)

        # 获取嵌入向量
        noc_embedded = noc_embedding(noc_indices)  # (batch_size, sequence_length, embedding_dim)  
        host_embedded = host_embedding(host_indices)  # (batch_size, sequence_length, embedding_dim)  

        # 拼接嵌入向量和数值特征
        combined_inputs = torch.cat((noc_embedded, host_embedded, numeric_inputs), dim=2)  # (batch_size, sequence_length, total_dim)  

        self.x_input_future = torch.tensor(combined_inputs, dtype=torch.float32)