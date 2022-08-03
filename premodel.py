from turtle import forward
import torch
import torch.nn as nn


class Qnet(nn.Module):
    def __init__(self, dim_in = 10, dim_out = 3) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(dim_in, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, dim_out)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x
    
        