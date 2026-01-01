import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, image_encoder, loc_encoder, decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.loc_encoder = decoder
        self.decoder = decoder

    def forward(self, image, raw_coord):
        encoded_image = self.image_encoder(image)
        encoded_coords = self.loc_encoder(raw_coord)
        encoded_interacting_image_coords = np.mul(encoded_image, encoded_coords) # vector pointwise multiplication
        category_index = self.decoder(encoded_interacting_image_coords)
        return category_index

class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, category_count):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, category_count)
        )

    def forward(self, embedding):
        return self.model(embedding)



