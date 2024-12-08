from abc import ABCMeta
from torch import nn, Tensor


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, K,  p_dropout=0.2, bias=True, bidirectional=True):
        super(LSTM, self).__init__()
        if bidirectional:
            bi_layers = 2
        else:
            bi_layers = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=p_dropout, bidirectional=bidirectional)
        self.layer1 = nn.Linear(int(K*hidden_size*bi_layers), int((K*hidden_size*bi_layers)/2))
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.layer2 = nn.Linear(int((K*hidden_size*bi_layers)/2), 1)
        

    def forward(self, x, prints=False):
        batch_size = x.shape[0]
        x = x.transpose(0,1)
        x, _ = self.lstm(x)
        x = x.transpose(0,1)
        x = x.reshape(batch_size, -1)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x