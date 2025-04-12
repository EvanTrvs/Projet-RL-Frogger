import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.dqn.networks.nn import NN

# Classe pour le modèle entièrement connecté
class FullyConnectedNN(NN):
    def __init__(self, input_size, output_size, dropout=0.3, additional_features_size=0):
        super(FullyConnectedNN, self).__init__(additional_features_size)

        # Couches fully connected
        self.fc1 = nn.Linear(input_size + additional_features_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(dropout)

    # Fonction forward pour le modèle FC
    # x = (batch_size, input_size) type torch.Tensor(float32)
    # additional_features = (batch_size, additional_features_size) type torch.Tensor(float32)
    def forward(self, x, additional_features=None):
        if additional_features is not None:
            additional_features = additional_features.float()
            x = torch.cat((x, additional_features), dim=1)  # Concaténation des features supplémentaires

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
