import torch
from torch import nn
from torch.autograd import Variable

# Define model
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):  # 添加output_size
        super(MyLSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size  # 新增

        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)  # 添加全连接层

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def forward(self, input):
        self.hidden = self.init_hidden(input.size(0))
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        
        # 将LSTM的输出传递给全连接层
        output = self.fc(lstm_out[:, -1, :])  # 使用最后一个时刻的输出
        
        return output
