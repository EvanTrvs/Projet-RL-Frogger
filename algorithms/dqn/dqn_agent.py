"""
Agent DQN principal qui gère l'interaction avec l'environnement et l'apprentissage.
Cette classe sert d'interface principale et délègue les responsabilités spécifiques
aux différents modules (réseau neuronal, mémoire, etc.).
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from common.utils.config import Config

from algorithms.dqn.networks.nn_manager import NNManager

from algorithms.dqn.memory.replay_prio import PrioritizedReplayBuffer
from algorithms.dqn.memory.replay import ReplayBuffer


# Classe pour l'agent DQN
class DQNAgent:
    def __init__(self, config: Config, input_shape, output_shape):
        """
        :param model_type: Type de modèle ('cnn' ou 'fc').
        :param input_shape: Taille de l'entrée (pour la structure CNN/FC).
        :param additional_features_size: Taille des features supplémentaires.
        :param learning_rate: Taux d'apprentissage.
        :param epsilon: Valeur initiale d'epsilon pour la stratégie epsilon-greedy.
        :param epsilon_decay: Taux de décroissance d'epsilon.
        :param epsilon_min: Valeur minimale d'epsilon.
        :param gamma: Facteur de réduction pour les récompenses futures.
        :param tau: Taux de mise à jour douce pour le modèle cible.
        :param buffer_size: Taille du replay buffer.
        :param batch_size: Taille du batch pour l'entraînement.
        :param use_prioritized_replay: Si True, utilise PrioritizedReplayBuffer. Sinon, utilise ReplayBuffer.
        :param output_shape: Shape de la sortie pour le modèle.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
            
        # Gestion des shapes input et output
        self.output_shape = output_shape
        # On transforme l'input shape en 3 dimensions si elle n'en a que 2
        self.input_shape = (input_shape[0], input_shape[1], 1) if len(input_shape) == 2 else input_shape

        # Initialisation des composants pour le NN
        self.model_manager = NNManager(config, self.input_shape, self.output_shape, self.device)        
        self.model = self.model_manager.model.to(self.device) # Déplacer le modèle principal
        self.target_model = NNManager(config, self.input_shape, self.output_shape).model.to(self.device) # Déplacer le modèle cible
        self.target_model.load_state_dict(self.model.state_dict()) # Assurer la synchronisation initiale après déplacement
        self.target_model.eval() # Mettre le modèle cible en mode évaluation
        
        # Initialisation de l'optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.dqn.hyperparameters.learning_rate)
        
        self.criterion = nn.SmoothL1Loss() if self.config.dqn.hyperparameters.loss_fn == "huber" else nn.MSELoss()
        
        # Initialisation des paramètres d'apprentissage
        self.steps_count = 0
        self.epsilon = 1
        self.epsilon_decay = self.config.dqn.hyperparameters.epsilon_decay
        self.epsilon_min = self.config.dqn.hyperparameters.minimum_epsilon
        self.gamma = self.config.dqn.hyperparameters.gamma
        self.tau = self.config.dqn.hyperparameters.tau
        self.batch_size = self.config.dqn.memory.batch_size
        self.grad_clip = self.config.dqn.hyperparameters.grad_clip

        # Initialisation du replay buffer en fonction du paramètre use_prioritized_replay
        if self.config.dqn.memory.type == "prioritized":
            self.replay_buffer = PrioritizedReplayBuffer(self.config, self.input_shape, self.output_shape, device=self.device)
        else:
            self.replay_buffer = ReplayBuffer(self.config, self.input_shape, self.output_shape, device=self.device)

    def select_action(self, state, features=None, eval=False):
        """
        Sélectionne une action en utilisant une stratégie epsilon-greedy.

        :param state: État actuel (image numpy array uint8).
        :param features: Features supplémentaires.
        :return: Indice de l'action sélectionnée.
        """
        if features is not None and self.config.dqn.network.num_features != len(features):
            raise ValueError(f"Le nombre de features fourni ({len(features)}) ne correspond pas au nombre de features défini dans le config ({self.config.dqn.network.num_features})")
        
        if np.random.rand() <= self.epsilon and not eval:
            return np.random.choice(self.output_shape)  # Action aléatoire

        # On ajoute le channel a la shape[2] si le state est un np.array 2D, pour avoir [height, width, channel]
        if len(state.shape) == 2:
            state = state.reshape(*state.shape, 1)

        q_values = self.model_manager.predict(state, features).to("cpu")  # Prédiction des Q-valeurs
        return torch.argmax(q_values, dim=1).item()
    
    # fonction pour le stockage des transitions dans le replay buffer
    def store_transition(self, state, action, reward, next_state, done, features=None, next_features=None):
        """
        Stockage d'une transition dans le replay buffer.
        """
        if features is not None and self.config.dqn.network.num_features != len(features):
            raise ValueError(f"Le nombre de features fourni ({len(features)}) ne correspond pas au nombre de features défini dans le config ({self.config.dqn.network.num_features})")
        
        if next_features is not None and self.config.dqn.network.num_features != len(next_features):
            raise ValueError(f"Le nombre de features fourni 'next'({len(next_features)}) ne correspond pas au nombre de features défini dans le config ({self.config.dqn.network.num_features})")
        
        # On ajoute le channel a la shape[2] si le state est un np.array 2D, pour avoir [height, width, channel]
        if len(state.shape) == 2:
            state = state.reshape(state.shape[0], state.shape[1], 1)
            next_state = next_state.reshape(next_state.shape[0], next_state.shape[1], 1)
        
        if self.config.dqn.network.network_type == 'cnn':
            state = torch.as_tensor(state, dtype=torch.float16) / 255.0 # Conversion et Normalisation des valeurs de 0 à 1
            next_state = torch.as_tensor(next_state, dtype=torch.float16) / 255.0
            state = state.permute(2, 0, 1) # Permutation [channels, height, width]
            next_state = next_state.permute(2, 0, 1)

        elif self.config.dqn.network.network_type == 'fc':
            # Conversion en tenseur PyTorch aplati, normalisé de 0 à 1, sans ajout de dimension de batch
            state = torch.as_tensor(state, dtype=torch.float16).view(-1) / 255.0
            next_state = torch.as_tensor(next_state, dtype=torch.float16).view(-1) / 255.0
            
            if features is not None:
                features = torch.as_tensor(features, dtype=torch.float16) # Conversion en tenseur PyTorch
                next_features = torch.as_tensor(next_features, dtype=torch.float16)
                state = torch.cat((state, features), dim=0) # Concaténation des features
                next_state = torch.cat((next_state, next_features), dim=0)
                features = next_features = None

        self.replay_buffer.add((state, action, reward, next_state, done, features, next_features))

    def train_on_batch(self):
        """
        Entraîne l'agent sur un batch d'expériences échantillonnées à partir du replay buffer.
        """
        if self.replay_buffer.real_size < self.batch_size + 1000:
            return None, None  # Ne pas entraîner si le replay buffer n'est pas suffisamment rempli

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            experiences, weights_cpu, tree_idxs = self.replay_buffer.sample(self.batch_size)
            weights = torch.as_tensor(weights_cpu, dtype=torch.float32, device=self.device).squeeze()
        else:
            experiences = self.replay_buffer.sample(self.batch_size)
            weights = None
            tree_idxs = None
        
        #    Utiliser torch.as_tensor pour créer ou convertir et placer sur le device
        states = torch.as_tensor(experiences[0], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(experiences[1], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(experiences[2], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(experiences[3], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(experiences[4], dtype=torch.bool, device=self.device)

        # Gérer les features optionnelles
        additional_features = None
        next_additional_features = None
        # Vérifier si les features sont présentes dans l'expérience ET si elles sont utilisées par le modèle
        if self.config.dqn.network.num_features > 0:
             if len(experiences) > 5 and experiences[5] is not None:
                 additional_features = torch.as_tensor(experiences[5], dtype=torch.float32, device=self.device)
             if len(experiences) > 6 and experiences[6] is not None:
                 next_additional_features = torch.as_tensor(experiences[6], dtype=torch.float32, device=self.device)

        # Ajout de print pour le débogage
        #print("States shape:", states.shape)
        #print("Actions shape:", actions.shape)
        #print("Rewards shape:", rewards.shape)
        #print("Next states shape:", next_states.shape)
        #print("Dones shape:", dones.shape)
        #print("Actions dtype:", actions.dtype)

        # 3. Forward Pass du modèle principal (sur self.device)
        self.model.train()
        q_values = self.model(states, additional_features)

        # 4. Forward Pass du modèle cible (sur self.device)
        with torch.no_grad():
            next_q_values = self.target_model(next_states, next_additional_features)
        
        # Ajout de print pour le débogage
        #print("Q-values shape:", q_values.shape)
        #print("Next Q-values shape:", next_q_values.shape)

        # 5. Calcul des Q-Cibles (sur self.device)
        max_next_q_values = torch.max(next_q_values, dim=1)[0] # shape: [batch_size]
        
        # Calculer les cibles : R + gamma * max_q'(s', a') * (non terminal)
        # (~dones) transforme les booléens True (fini) en 0 et False (pas fini) en 1
        q_targets_next = rewards + (self.gamma * max_next_q_values * (~dones))
        
        
        # Mettre à jour q_targets seulement pour les actions réellement prises
        # actions shape est [batch_size] ou [batch_size, 1]. gather attend [batch_size, 1]
        # L'indexation avancée attend [batch_size] pour les lignes et [batch_size] pour les colonnes (actions)
        if actions.dim() == 1:
             actions_for_indexing = actions
        else:
             actions_for_indexing = actions.squeeze(1) # Enlever la dim 1 si elle existe
             
        # Extraire les Q-valeurs prédites POUR LES ACTIONS PRISES
        # q_values est (batch_size, num_actions)
        # On indexe avec [range(batch_size), actions_prises] pour obtenir les Q-valeurs spécifiques
        q_pred_action = q_values[torch.arange(self.batch_size, device=self.device), actions_for_indexing]
        
        # 6. Calcul de la Perte (sur self.device)
        #    La perte est calculée entre la prédiction pour l'action prise et sa cible q_targets_next
        if weights is not None:
            # Utiliser un criterion SANS réduction pour appliquer les poids correctement
            criterion_no_reduction = nn.SmoothL1Loss(reduction='none') if self.config.dqn.hyperparameters.loss_fn == "huber" else nn.MSELoss(reduction='none')
            # Calculer la perte élément par élément pour l'action prise
            loss_per_item_taken = criterion_no_reduction(q_pred_action, q_targets_next) # Shape: (batch_size,)
            # Appliquer les poids IS (element-wise multiplication)
            weighted_loss = loss_per_item_taken * weights # Shape: (batch_size,) * (batch_size,) -> (batch_size,)
            # Calculer la perte finale comme moyenne des pertes pondérées
            loss = weighted_loss.mean() # Scalaire
        else:
            # Si pas de PER, utiliser le criterion défini (qui peut avoir reduction='mean')
            # Note: self.criterion doit être utilisé ici, pas criterion_no_reduction
            loss = self.criterion(q_pred_action, q_targets_next) # Calcul direct de la perte (moyennée ou non selon self.criterion)
            # S'assurer que c'est un scalaire si ce n'est pas déjà fait par le criterion
            if loss.dim() != 0:
                 loss = loss.mean()
        
        

        # 7. Rétropropagation et Mise à jour (sur self.device)
        self.optimizer.zero_grad()
        loss.backward() # Calcule les gradients sur self.device

        # Optionnel: Clipping des gradients
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step() # Met à jour les poids du modèle sur self.device
        
        # 8. Calcul de l'Erreur TD pour PER (calcul sur device, transfert vers CPU)
        with torch.no_grad():
             # Erreur absolue entre la prédiction pour l'action prise et sa cible
             td_error_tensor = torch.abs(q_pred_action - q_targets_next) # Shape: (batch_size,)
        td_error = td_error_tensor.detach().cpu().numpy() # numpy array, shape: (batch_size,)

        # 9. Mise à jour d'epsilon (sur CPU)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 10. Mise à jour des priorités (sur CPU, utilise td_error numpy)
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priorities(tree_idxs, td_error)

        # 11. Retourner la perte (scalaire CPU) et l'erreur TD (numpy CPU)
        return loss.item(), td_error

    def update_target_model(self):
        """
        Met à jour le modèle cible.

        :param hard_update: Si True, effectue une mise à jour rigide. Sinon, effectue une mise à jour douce.
        """
        if not self.config.dqn.hyperparameters.soft_update:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        """
        Sauvegarde l'agent DQN complet dans un fichier.

        :param path: Chemin où sauvegarder l'agent.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'gamma': self.gamma,
            'tau': self.tau,
            'steps_count': self.steps_count,
            'replay_buffer': self.replay_buffer.save()
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """
        Charge l'agent DQN complet à partir d'un fichier.

        :param path: Chemin où charger l'agent.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.epsilon_min = checkpoint['epsilon_min']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.steps_count = checkpoint['steps_count']
        self.replay_buffer.load(checkpoint['replay_buffer'])
        self.model.eval()
        self.target_model.eval()
