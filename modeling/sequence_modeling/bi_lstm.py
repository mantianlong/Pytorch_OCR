import torch.nn as nn
import torch
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate = nn.Linear(input_size + hidden_size, cell_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        forget_gate = self.gate(combined)
        input_gate = self.gate(combined)
        output_gate = self.gate(combined)
        forget_gate = self.sigmoid(forget_gate)
        input_gate = self.sigmoid(input_gate)
        output_gate = self.sigmoid(output_gate)
        cell_helper = self.gate(combined)
        cell_helper = self.tanh(cell_helper)
        cell = torch.add(torch.mul(cell, forget_gate), torch.mul(cell_helper, input_gate))
        hidden = torch.mul(self.tanh(cell), output_gate)
