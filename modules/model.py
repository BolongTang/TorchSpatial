import torch.nn as nn
import numpy as np

class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, category_count):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, category_count)
        )

    def forward(self, embedding):
        return self.model(embedding)



