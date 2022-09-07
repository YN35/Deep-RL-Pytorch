
import torch
import torch.nn as nn

def get_model_class(name):
    return getattr(getattr(__import__('mllib'), 'model'), name)

class DQN_network_CartPole_v1(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(4, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = torch.nn.Linear(4, 2)
        
        self.relu = nn.ReLU()
        self.softplus = torch.nn.Softplus(beta=100)
        self.sigmoid = torch.nn.Sigmoid()
        
        # init_weights(self)
        
    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x