# Ce code est basé sur l'implémentation https://github.com/Howuhh/prioritized_experience_replay/tree/main
# Créé par Howuhh (Alexander Nikulin) et AlexPasqua (Alex Pasquali).
# Basé sur l'article : Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. ArXiv:1511.05952 [Cs]. http://arxiv.org/abs/1511.05952

import torch
import random
import numpy as np

from common.utils.config import Config

from algorithms.dqn.memory.sumtree import SumTree

# Classe PrioritizedReplayBuffer
# Cette classe implémente un buffer de relecture priorisé pour stocker les transitions observées par un agent.
# Les transitions sont échantillonnées en fonction de leurs priorités, ce qui permet d'améliorer l'efficacité de l'apprentissage.
class PrioritizedReplayBuffer:
    def __init__(self, config: Config, input_shape, output_size, device="cpu"):
        #state_size, action_size, buffer_size, feature_size=None, eps=1e-2, alpha=0.1, beta=0.1, device="cpu"):
        """
        Initialise le buffer de relecture priorisé.

        :param state_size: Taille de l'état.
        :param action_size: Taille de l'action.
        :param buffer_size: Taille maximale du buffer.
        :param feature_size: Taille des features (optionnel).
        :param eps: Priorité minimale pour éviter les probabilités nulles.
        :param alpha: Paramètre de priorisation.
        :param beta: Paramètre de correction de l'importance-sampling.
        """
        #print(f"memory type: {self.config.dqn.memory.type} - network type: {self.config.dqn.network.network_type} - input_shape: {input_shape}")
        self.config = config
        self.buffer_size = self.config.dqn.memory.buffer_size
        self.device = device
        self.tree = SumTree(size=self.buffer_size)
        
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
            
        # Paramètres de la priorisation
        self.eps = self.config.dqn.memory.prior_eps  # Priorité minimale
        self.alpha = self.config.dqn.memory.alpha  # Détermine le niveau de priorisation
        self.beta = self.config.dqn.memory.beta  # Détermine le niveau de correction de l'importance-sampling
        self.max_priority = self.config.dqn.memory.prior_eps  # Priorité initiale pour les nouveaux échantillons

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

        # Stocke l'index de la transition avec la priorité maximale dans le SumTree
        self.tree.add(self.max_priority, self.count)

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
        :return: Batch de transitions, poids d'importance-sampling, index des transitions dans le SumTree.
        """
        #assert self.real_size >= batch_size, "Le buffer contient moins d'échantillons que la taille du batch"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float32)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        # Divise l'intervalle [0, p_total] en k intervalles égaux pour échantillonner un mini-batch de taille k
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            
            # Réessayer si sample_idx est None
            attempts = 0
            while sample_idx is None and attempts < 5:
                #print(f"sample_idx is None, attempts: {attempts}")
                cumsum = random.uniform(a, b)
                tree_idx, priority, sample_idx = self.tree.get(cumsum)
                attempts += 1

            priorities[i] = torch.as_tensor(priority, dtype=torch.float32)
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta
        
        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

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
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        """
        Met à jour les priorités des transitions dans le SumTree.

        :param data_idxs: Index des transitions dans le buffer.
        :param priorities: Nouvelles priorités des transitions.
        """
        # Assurez-vous que les priorités sont de type numpy
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
            
        #assert priorities.shape == (len(data_idxs),), "Priorities should have the same shape as data_idxs"

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
            
    def save(self):
        """
        Sauvegarde le contenu du buffer de relecture priorisé.

        :return: Dictionnaire contenant les données du buffer.
        """
        return {
            'tree': self.tree.save(),
            'eps': self.eps,
            'alpha': self.alpha,
            'beta': self.beta,
            'max_priority': self.max_priority,
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
        Charge le contenu du buffer de relecture priorisé.

        :param data: Dictionnaire contenant les données du buffer.
        """
        self.tree.load(data['tree'])
        self.eps = data['eps']
        self.alpha = data['alpha']
        self.beta = data['beta']
        self.max_priority = data['max_priority']
        self.state = data['state']
        self.action = data['action']
        self.reward = data['reward']
        self.next_state = data['next_state']
        self.done = data['done']
        self.count = data['count']
        self.real_size = data['real_size']