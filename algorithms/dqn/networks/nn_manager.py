import torch
import numpy as np

from common.utils.config import Config

from algorithms.dqn.networks.convolutional_nn import ConvolutionalNN
from algorithms.dqn.networks.fully_connected_nn import FullyConnectedNN

# Classe pour gérer les modèles
class NNManager:
    def __init__(self, config: Config, input_shape, output_size, device="cpu"):
        self.config = config
        self.device = device
        self.model_type = config.dqn.network.network_type
        self.additional_features_size = config.dqn.network.num_features

        if self.model_type == 'cnn':
            if input_shape is None:
                raise ValueError("input_shape must be provided for CNN model")
            self.model = ConvolutionalNN(input_shape, output_size, self.config.dqn.network.dropout, self.additional_features_size)
        elif self.model_type == 'fc':
            if input_shape is None:
                raise ValueError("input_size must be provided for FC model")
            input_size = input_shape[0] * input_shape[1] * input_shape[2] + self.additional_features_size
            self.model = FullyConnectedNN(input_size, output_size, self.config.dqn.network.dropout, 0)
        else:
            raise ValueError("Invalid model_type. Choose 'cnn' or 'fc'.")

    # Traite le state et les features pour le modèle et retourne les q_values
    def predict(self, state, features=None):
        
        if self.config.dqn.network.network_type == 'cnn':
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device) / 255.0 # Conversion et Normalisation des valeurs de 0 à 1
            state = state.permute(2, 0, 1) # Permutation [channels, height, width]
            state = state.unsqueeze(0) # Ajout de la dimension batch

            if features is not None:        
                features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
                features = features.unsqueeze(0) # Ajout de la dimension batch

        elif self.model_type == 'fc':
            # On aplatit les images en tensor float32, avec normalisation de 0 à 1, et on ajoute la dimension batch (oneliner)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1) / 255.0  # Ajout de la dimension batch et aplatissement

            if features is not None:
                features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
                features = features.unsqueeze(0)  # Ajout de la dimension batch
                state = torch.cat((state, features), dim=1)  # Concaténation des features
                features = None

        # Mise du modèle en mode évaluation
        self.model.eval()
        with torch.no_grad():
            # Prédiction sans suivi des gradients
            output = self.model.forward(state, features)
        return output