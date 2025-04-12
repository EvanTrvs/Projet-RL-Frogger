# Ce code est basé sur l'implémentation https://github.com/Howuhh/prioritized_experience_replay/tree/main
# Créé par Howuhh (Alexander Nikulin) et AlexPasqua (Alex Pasquali).
# Basé sur l'article : Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. ArXiv:1511.05952 [Cs]. http://arxiv.org/abs/1511.05952

# Classe SumTree
# Cette classe implémente une structure de données en forme d'arbre binaire où chaque nœud interne stocke la somme des valeurs de ses enfants.
# Les feuilles de l'arbre stockent les priorités des transitions. Cette structure permet des mises à jour et des échantillonnages efficaces en O(log N).
class SumTree:
    # The ‘sum-tree’ data structure used here is very similar in spirit to the array representation
    # of a binary heap. However, instead of the usual heap property, the value of a parent node is
    # the sum of its children. Leaf nodes store the transition priorities and the internal nodes are
    # intermediate sums, with the parent node containing the sum over all priorities, p_total. This
    # provides a efficient way of calculating the cumulative sum of priorities, allowing O(log N) updates
    # and sampling. (Appendix B.2.1, Proportional prioritization)

    # Additional useful links
    # Good tutorial about SumTree data structure:  https://adventuresinmachinelearning.com/sumtree-introduction-python/
    # How to represent full binary tree as array: https://stackoverflow.com/questions/8256222/binary-tree-represented-using-array
    def __init__(self, size):
        """
        Initialise le SumTree.

        :param size: Taille maximale du SumTree.
        """
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        """
        Retourne la somme totale des priorités stockées dans le SumTree.
        """
        return self.nodes[0]

    def update(self, data_idx, value):
        """
        Met à jour la priorité d'une transition dans le SumTree.

        :param data_idx: Index de la transition dans le buffer.
        :param value: Nouvelle priorité de la transition.
        """
        idx = data_idx + self.size - 1  # Index de l'enfant dans le tableau de l'arbre
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        """
        Ajoute une nouvelle transition avec une priorité donnée dans le SumTree.

        :param value: Priorité de la nouvelle transition.
        :param data: Index de la nouvelle transition dans le buffer.
        """
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        """
        Récupère l'index et la priorité d'une transition en fonction d'une somme cumulative.

        :param cumsum: Somme cumulative pour échantillonner une transition.
        :return: Index de la transition dans le buffer, priorité de la transition, index de la transition dans le SumTree.
        """
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        """
        Représentation en chaîne de caractères du SumTree.
        """
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"
    
    def save(self):
        """
        Sauvegarde le contenu du SumTree.

        :return: Dictionnaire contenant les données du SumTree.
        """
        return {
            'nodes': self.nodes,
            'data': self.data,
            'count': self.count,
            'real_size': self.real_size
        }

    def load(self, data):
        """
        Charge le contenu du SumTree.

        :param data: Dictionnaire contenant les données du SumTree.
        """
        self.nodes = data['nodes']
        self.data = data['data']
        self.count = data['count']
        self.real_size = data['real_size']