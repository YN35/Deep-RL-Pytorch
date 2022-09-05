
import torch
import torch.nn as nn

def get_model_class(name):
    return getattr(getattr(__import__('mllib'), 'model'), name)

class DQN_network_orig(nn.Module):
    def __init__(
        self, pos_enc, dim_emb_position=3, dim_emb=512, normalization=None, activation=None,
    ):
        super().__init__()
        
        self.pos_enc = pos_enc
        self.fc1 = torch.nn.Linear(dim_emb_position, dim_emb)
        self.fc2 = torch.nn.Linear(dim_emb, dim_emb)
        self.fc3 = torch.nn.Linear(dim_emb, dim_emb)
        self.fc4 = torch.nn.Linear(dim_emb, dim_emb)
        self.fc5 = torch.nn.Linear(dim_emb, 3)
        
        self.relu = nn.ReLU()
        self.softplus = torch.nn.Softplus(beta=100)
        self.sigmoid = torch.nn.Sigmoid()
        
        # init_weights(self)
        
    def forward(self, x):
        x = self.pos_enc(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        # x = self.softplus(self.fc5(x))
        x = self.sigmoid(self.fc5(x))
        return x