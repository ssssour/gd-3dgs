
import torch
import torch.nn as nn

class GaussianPredictorMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):  # Change input_dim to 3
        super(GaussianPredictorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.fc1(x)
        
        x = self.relu(x)
        
        x = self.fc2(x)
        
        x = self.relu(x)
        
        x = self.fc3(x)
        
        return x