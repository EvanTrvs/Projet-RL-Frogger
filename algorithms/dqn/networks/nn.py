import torch.nn as nn

# Classe de base pour les modèles
class NN(nn.Module):
    def __init__(self, additional_features_size=0):
        super(NN, self).__init__()
        self.additional_features_size = additional_features_size

    def forward(self, x, additional_features=None):
        raise NotImplementedError("Forward method must be implemented by subclasses")