import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from algorithms.dqn.networks.nn import NN

# Classe pour le modèle CNN simplifié, avec possibilité d'hybrid avec des features supplémentaires
class ConvolutionalNN(NN):
    def __init__(self, input_shape, output_size, dropout=0.3, additional_features_size=0):
        super(ConvolutionalNN, self).__init__(additional_features_size)

        # input_shape est (height, width, channels)
        # On réorganise pour PyTorch (channels, height, width)
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        in_channels = self.input_shape[0]

        # Couches convolutives
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(dropout)

        # Calcul de la taille de sortie des couches convolutives
        conv_output_size = self._calculate_conv_output_size()

        # Couches fully connected
        self.fc1 = nn.Linear(conv_output_size + additional_features_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def _calculate_conv_output_size(self):
        # Calcul de la taille de sortie des couches convolutives en passant un tenseur factice
        x = torch.zeros(1, *self.input_shape)  # (batch_size, channels, height, width)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return int(np.prod(x.size()))

    # Fonction forward pour le modèle CNN
    # x = (batch_size, in_channels, height, width) type torch.Tensor(float32)
    # additional_features = (batch_size, additional_features_size) type torch.Tensor(float32)
    def forward(self, x, additional_features=None):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Aplatissement du tenseur

        if additional_features is not None:
            additional_features = additional_features.float()
            x = torch.cat((x, additional_features), dim=1)  # Concaténation des features supplémentaires

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
