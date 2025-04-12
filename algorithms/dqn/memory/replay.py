# Ce code est basé sur l'implémentation https://github.com/Howuhh/prioritized_experience_replay/tree/main
# Créé par Howuhh (Alexander Nikulin) et AlexPasqua (Alex Pasquali).
# Basé sur l'article : Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. ArXiv:1511.05952 [Cs]. http://arxiv.org/abs/1511.05952

import torch
import numpy as np

from common.utils.config import Config

# Classe ReplayBuffer
# Cette classe implémente un buffer de relecture simple pour stocker les transitions observées par un agent.
# Les transitions sont échantillonnées de manière uniforme.
class ReplayBuffer:
    def __init__(self, config: Config, input_shape, output_size, device="cpu"):
        # state_size, action_size, buffer_size, feature_size=None, device="cpu"):
        """
        Initialise le buffer de relecture.

        :param state_size: Taille de l'état.
        :param action_size: Taille de l'action.
        :param buffer_size: Taille maximale du buffer.
        :param feature_size: Taille des features (optionnel).
        """
        #print(f"memory type: {self.config.dqn.memory.type} - network type: {self.config.dqn.network.network_type} - input_shape: {input_shape}")
        self.config = config
        self.buffer_size = self.config.dqn.memory.buffer_size
        self.device = device
        
        # Si le réseau est de type FC, on ne considère pas les features (directement dans l'état flatten)
        if self.config.dqn.network.network_type != "fc":
            # Si CNN et features > 0, on considère les features dans transition[5], mais si CNN et features = 0, features = None
            self.feature_size = self.config.dqn.network.num_features if self.config.dqn.network.num_features > 0 else None
            input_shape = (input_shape[2], input_shape[0], input_shape[1])
            self.state = torch.empty(self.buffer_size, *input_shape, dtype=torch.float16)
            self.next_state = torch.empty(self.buffer_size, *input_shape, dtype=torch.float16)
        else:
            # FC features = None et input_shape = (input_shape[2], input_shape[0], input_shape[1])
            self.feature_size = None
            input_shape = (input_shape[2] * input_shape[0] * input_shape[1]) + self.config.dqn.network.num_features
            self.state = torch.empty(self.buffer_size, input_shape, dtype=torch.float16)
            self.next_state = torch.empty(self.buffer_size, input_shape, dtype=torch.float16)
            
        self.action = torch.empty(self.buffer_size, dtype=torch.int64)
        self.reward = torch.empty(self.buffer_size, dtype=torch.float32)
        self.done = torch.empty(self.buffer_size, dtype=torch.bool)
        
        # Features et next_features (optionnel)
        self.use_features = self.feature_size is not None
        if self.use_features:
            self.features = torch.empty(self.buffer_size, self.feature_size, dtype=torch.float32)
            self.next_features = torch.empty(self.buffer_size, self.feature_size, dtype=torch.float32)

        self.max_priority = None
        self.count = 0
        self.real_size = 0

    # si fc, features flatten dans l'état
    # si cnn, features dans transition[5]
    def add(self, transition):
        """
        Ajoute une transition au buffer.

        :param transition: Tuple contenant (état, action, récompense, état suivant, terminé, [features, next_features]).
        """
        state, action, reward, next_state, done = transition[:5]
        features = next_features = None
        if self.use_features:
            features, next_features = transition[5], transition[6]

        # Stocke la transition dans le buffer
        self.state[self.count] = state
        self.action[self.count] = torch.as_tensor(action, dtype=torch.int64)
        self.reward[self.count] = torch.as_tensor(reward, dtype=torch.float32)
        self.next_state[self.count] = next_state
        self.done[self.count] = torch.as_tensor(done, dtype=torch.bool)

        if self.use_features:
            self.features[self.count] = torch.as_tensor(features, dtype=torch.float32)
            self.next_features[self.count] = torch.as_tensor(next_features, dtype=torch.float32)

        # Met à jour les compteurs
        self.count = (self.count + 1) % self.buffer_size
        self.real_size = min(self.buffer_size, self.real_size + 1)

    def sample(self, batch_size):
        """
        Échantillonne un batch de transitions à partir du buffer.

        :param batch_size: Taille du batch à échantillonner.
        :return: Batch de transitions.
        """
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        with torch.no_grad():
            batch = (
                self.state[sample_idxs],
                self.action[sample_idxs],
                self.reward[sample_idxs],
                self.next_state[sample_idxs],
                self.done[sample_idxs]
            )
            if self.use_features:
                batch += (
                    self.features[sample_idxs],
                    self.next_features[sample_idxs]
                )
        return batch
    
    def save(self):
        """
        Sauvegarde le contenu du buffer de relecture.

        :return: Dictionnaire contenant les données du buffer.
        """
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'count': self.count,
            'real_size': self.real_size
        }

    def load(self, data):
        """
        Charge le contenu du buffer de relecture.

        :param data: Dictionnaire contenant les données du buffer.
        """
        self.state = data['state']
        self.action = data['action']
        self.reward = data['reward']
        self.next_state = data['next_state']
        self.done = data['done']
        self.count = data['count']
        self.real_size = data['real_size']