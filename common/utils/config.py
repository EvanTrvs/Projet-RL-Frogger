import json
import os
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, List

@dataclass
class EnvConfig:
    """
    Configuration de l'environnement.

    Cette classe définit les paramètres spécifiques à l'environnement Atari utilisé pour l'entraînement.

    Paramètres :
    - name (str) : Nom de l'environnement Atari (ex: "ALE/Frogger-v5").
    - render_mode (str) : Mode de rendu de l'environnement (ex: "rgb_array").
    - max_episode_steps (Optional[int]) : Nombre maximum de pas par épisode (None pour pas de limite).
    - reward_shaping (Optional[str]) : Type de mise en forme des récompenses (None pour pas de mise en forme).
    - EpisodicLifeEnv (bool) : Si True, met fin à un épisode lorsque l'agent perd une vie.
    """
    name: str = "ALE/Frogger-v5"
    render_mode: str = "rgb_array"
    max_episode_steps: Optional[int] = 500
    reward_shaping: Optional[str] = "classic"
    EpisodicLifeEnv: bool = True

    def __post_init__(self):
        if self.reward_shaping not in [None, "classic", "custom"]:
            raise ValueError("reward_shaping doit être 'classic' ou 'custom'")

@dataclass
class PreprocessingConfig:
    """
    Configuration du prétraitement.

    Cette classe définit les paramètres de prétraitement des observations de l'environnement.

    Paramètres :
    - rognage_bords (bool) : Si True, rogne les bords de l'image.
    - grayscale (bool) : Si True, convertit les images en niveaux de gris.
    - flou (List[int]) : Taille des noyaux pour appliquer un flou gaussien.
    - downscaling (List[int]) : Taille de redimensionnement des images.
    - centrer_grenouille (List[int]) : Paramètres pour centrer la grenouille dans l'image.
    - mappage_pixels (bool) : Si True, applique un mappage de pixels custom.
    - frame_stacking (int) : Nombre de frames empilées.
    - frame_skip (str) : Type de frame skip ("first-mid-last", "none", "first-last").
    - extract_features (bool) : Si True, extrait des features spécifiques des observations.
    """
    rognage_bords: bool = True
    grayscale: bool = True
    flou: List[int] = field(default_factory=lambda: [0, 0])
    downscaling: List[int] = field(default_factory=lambda: [72, 84])
    centrer_grenouille: List[int] = field(default_factory=lambda: [0, 0, [0, 0, 0]])
    mappage_pixels: bool = False
    frame_stacking: int = 1
    frame_skip: str = "none"
    extract_features: bool = False

    def __post_init__(self):
        if self.frame_skip not in ["first-mid-last", "none", "first-last"]:
            raise ValueError("frame_skip doit être 'first-mid-last', 'none' ou 'first-last'")
        if not (0 <= self.frame_stacking <= 10):
            raise ValueError("frame_stacking doit être entre 0 et 10")

@dataclass
class DQNHyperparametersConfig:
    """
    Configuration des hyperparamètres pour l'entraînement DQN.

    Cette classe définit les hyperparamètres utilisés pour l'entraînement de l'agent DQN.

    Paramètres :
    - epsilon_decay (float) : Taux de décroissance d'epsilon.
    - minimum_epsilon (float) : Valeur minimale d'epsilon.
    - learning_rate (float) : Taux d'apprentissage.
    - target_update (int) : Fréquence de mise à jour du réseau cible.
    - gamma (float) : Facteur de discount.
    - soft_update (bool) : Si True, utilise une mise à jour douce du réseau cible.
    - tau (float) : Taux de mise à jour douce.
    - optimizer (str) : Optimiseur utilisé ("adam").
    - loss_fn (str) : Fonction de perte ("mse", "huber").
    - grad_clip (float) : Valeur de clipping des gradients.
    - steps_frequency_update (int) : Fréquence de mise à jour des steps.
    """
    epsilon_decay: float = 0.9995
    minimum_epsilon: float = 0.05
    learning_rate: float = 0.00025
    target_update: int = 1000
    gamma: float = 0.99
    soft_update: bool = True
    tau: float = 0.01
    optimizer: str = "adam"
    loss_fn: str = "mse"
    grad_clip: float = 2.0
    steps_frequency_update: int = 2

    def __post_init__(self):
        if self.optimizer not in ["adam"]:
            raise ValueError("optimizer doit être 'adam'")
        if self.loss_fn not in ["mse", "huber"]:
            raise ValueError("loss_fn doit être 'mse' ou 'huber'")

@dataclass
class NetworkConfig:
    """
    Configuration du réseau neuronal.

    Cette classe définit les paramètres du réseau neuronal utilisé par l'agent DQN.

    Paramètres :
    - network_type (str) : Type de réseau ("cnn", "fc").
    - num_features (int) : Nombre de features numériques supplémentaires.
    - dropout (float) : Taux de dropout global.
    """
    network_type: str = "fc"
    num_features: int = 0
    dropout: float = 0.3

    def __post_init__(self):
        if self.network_type not in ["cnn", "fc"]:
            raise ValueError("network_type doit être 'cnn' ou 'fc'")

@dataclass
class MemoryConfig:
    """
    Configuration de la mémoire de replay.

    Cette classe définit les paramètres de la mémoire de replay utilisée pour l'entraînement de l'agent DQN.

    Paramètres :
    - type (str) : Type de replay buffer ("prioritized", "classic").
    - buffer_size (int) : Taille du replay buffer.
    - batch_size (int) : Taille des mini-batch.
    - alpha (float) : Paramètre de priorisation.
    - beta (float) : Paramètre de correction de biais.
    - prior_eps (float) : Petite constante pour éviter les priorités nulles.
    """
    type: str = "classic"
    buffer_size: int = 30000
    batch_size: int = 32
    alpha: float = 0.6
    beta: float = 0.4
    prior_eps: float = 1e-6

    def __post_init__(self):
        if self.type not in ["prioritized", "classic"]:
            raise ValueError("type doit être 'prioritized' ou 'classic'")

@dataclass
class DQNConfig:
    """
    Configuration complète pour le DQN.

    Cette classe regroupe les configurations des hyperparamètres, du réseau et de la mémoire pour l'agent DQN.

    Paramètres :
    - hyperparameters (DQNHyperparametersConfig) : Configuration des hyperparamètres.
    - network (NetworkConfig) : Configuration du réseau neuronal.
    - memory (MemoryConfig) : Configuration de la mémoire de replay.
    """
    hyperparameters: DQNHyperparametersConfig = field(default_factory=DQNHyperparametersConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

@dataclass
class TRPOHyperparametersConfig:
    """
    Configuration des hyperparamètres pour l'entraînement TRPO.

    Cette classe définit les hyperparamètres utilisés pour l'entraînement de l'agent TRPO.

    Paramètres :
    - max_kl (float) : Limite supérieure pour la divergence KL.
    - damping (float) : Paramètre d'amortissement.
    - l2_reg (float) : Régularisation L2.
    - use_fim (bool) : Si True, utilise la matrice d'information de Fisher.
    - gamma (float) : Facteur de discount.
    - tau (float) : Paramètre de mise à jour.
    - batch_size (int) : Taille des mini-batchs.
    """
    max_kl: float = 1e-2
    damping: float = 1e-2
    l2_reg: float = 1e-3
    use_fim: bool = True
    gamma: float = 0.99
    tau: float = 0.95
    batch_size: int = 32

@dataclass
class TRPOConfig:
    """
    Configuration complète pour le TRPO.

    Cette classe regroupe les configurations des hyperparamètres pour l'agent TRPO.

    Paramètres :
    - hyperparameters (TRPOHyperparametersConfig) : Configuration des hyperparamètres.
    """
    hyperparameters: TRPOHyperparametersConfig = field(default_factory=TRPOHyperparametersConfig)

@dataclass
class TrainingConfig:
    """
    Configuration de l'entraînement.

    Cette classe définit les paramètres de l'entraînement.

    Paramètres :
    - max_training_steps (int) : Nombre maximum de steps d'entraînement.
    - eval_interval_steps (int) : Intervalle de steps pour l'évaluation.
    - log_every_n_episodes (int) : Fréquence de logging des épisodes.
    - dual_model_road_river (bool) : Si True, entraîne deux modèles pour les deux zones différentes de l'environnement.
    """
    max_training_steps: int = 200000
    eval_interval_steps: int = 50000
    log_every_n_episodes: int = 5
    dual_model_road_river: bool = True

@dataclass
class EvaluateConfig:
    """
    Configuration de l'évaluation.

    Cette classe définit les paramètres de l'évaluation.

    Paramètres :
    - save_checkpoint (bool) : Si True, sauvegarde les checkpoints de poids du modèle.
    - num_parties (int) : Nombre d'épisodes d'évaluation.
    - env_gif (bool) : Si True, enregistre des GIFs des épisodes d'évaluation du jeu brut.
    - observation_gif (bool) : Si True, enregistre des GIFs des épisodes d'évaluation du jeu prétraité.
    - trace_actions (bool) : Si True, écrit un CSV de trace détaillé des steps des épisodes d'évaluation.
    - gif_fps (int) : FPS des GIFs.
    """
    save_checkpoint: bool = False
    num_parties: int = 5
    env_gif: bool = True
    observation_gif: bool = False
    trace_actions: bool = False
    gif_fps: int = 2

@dataclass
class Config:
    """
    Configuration complète pour l'entraînement et l'évaluation.

    Cette classe regroupe toutes les configurations nécessaires pour l'entraînement et l'évaluation de l'agent.

    Paramètres :
    - name (str) : Nom de l'expérience.
    - algorithm (str) : Algorithme utilisé ("dqn", "trpo").
    - output_dir (str) : Répertoire où les résultats seront sauvegardés.
    - device (str) : Dispositif d'exécution ("cpu", "cuda").
    - seed (Optional[int]) : Graine aléatoire pour la reproductibilité.
    - env (EnvConfig) : Configuration de l'environnement.
    - preprocessing (PreprocessingConfig) : Configuration du prétraitement.
    - dqn (DQNConfig) : Configuration du DQN.
    - trpo (TRPOConfig) : Configuration du TRPO.
    - training (TrainingConfig) : Configuration de l'entraînement.
    - evaluate (EvaluateConfig) : Configuration de l'évaluation.
    """
    name: str = "DefaultExperiment"
    algorithm: str = "dqn"
    output_dir: str = "results"
    device: str = "cpu"
    seed: Optional[int] = None
    env: EnvConfig = field(default_factory=EnvConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    trpo: TRPOConfig = field(default_factory=TRPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Crée une instance de Config à partir d'un dictionnaire.

        Args:
            config_dict (Dict[str, Any]): Dictionnaire contenant la configuration.

        Returns:
            Config: Instance de Config créée à partir du dictionnaire.
        """
        config_dict = config_dict.copy()

        # Filtrer les clés inconnues au niveau principal
        valid_fields = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}

        # Extraction des configurations spécifiques
        env_config = filtered_config.pop('env', {})
        preprocessing_config = filtered_config.pop('preprocessing', {})
        dqn_config = filtered_config.pop('dqn', {})
        trpo_config = filtered_config.pop('trpo', {})
        training_config = filtered_config.pop('training', {})
        evaluate_config = filtered_config.pop('evaluate', {})

        # Création des sous-configurations avec gestion des clés inconnues
        env = cls._create_instance(EnvConfig, env_config)
        preprocessing = cls._create_instance(PreprocessingConfig, preprocessing_config)
        dqn = DQNConfig(
            hyperparameters=cls._create_instance(DQNHyperparametersConfig, dqn_config.get('hyperparameters', {})),
            network=cls._create_instance(NetworkConfig, dqn_config.get('network', {})),
            memory=cls._create_instance(MemoryConfig, dqn_config.get('memory', {}))
        )
        trpo = TRPOConfig(
            hyperparameters=cls._create_instance(TRPOHyperparametersConfig, trpo_config.get('hyperparameters', {}))
        )
        training = cls._create_instance(TrainingConfig, training_config)
        evaluate = cls._create_instance(EvaluateConfig, evaluate_config)

        # Création de l'instance principale avec toutes les configurations
        return cls(
            **filtered_config,
            env=env,
            preprocessing=preprocessing,
            dqn=dqn,
            trpo=trpo,
            training=training,
            evaluate=evaluate
        )

    @staticmethod
    def _create_instance(cls, config_dict):
        """
        Crée une instance d'une classe à partir d'un dictionnaire en ignorant les clés inconnues.

        Args:
            cls (Type): Classe à instancier.
            config_dict (Dict[str, Any]): Dictionnaire contenant les paramètres de la classe.

        Returns:
            Instance de la classe créée à partir du dictionnaire.
        """
        # Filtrer les clés inconnues
        valid_fields = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_config)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """
        Charge la configuration depuis un fichier JSON.

        Args:
            path (str): Chemin vers le fichier JSON.

        Returns:
            Config: Instance de Config créée à partir du fichier JSON.
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration en dictionnaire.

        Returns:
            Dict[str, Any]: Dictionnaire représentant la configuration.
        """
        return {
            'name': self.name,
            'algorithm': self.algorithm,
            'output_dir': self.output_dir,
            'device': self.device,
            'seed': self.seed,
            'env': self.env.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'dqn': {
                'hyperparameters': self.dqn.hyperparameters.__dict__,
                'network': self.dqn.network.__dict__,
                'memory': self.dqn.memory.__dict__
            },
            'trpo': {
                'hyperparameters': self.trpo.hyperparameters.__dict__,
            },
            'training': self.training.__dict__,
            'evaluate': self.evaluate.__dict__
        }

    def save(self, save_dir: str):
        """
        Sauvegarde la configuration dans un fichier JSON.

        Args:
            save_dir (str): Répertoire où sauvegarder le fichier JSON.
        """
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def display(self):
        """
        Affiche la configuration de manière détaillée et compacte.
        """
        def format_dict(d, indent=0):
            items = []
            for k, v in d.items():
                if isinstance(v, dict):
                    items.append(f"{' ' * indent}{k}: {{")
                    items.extend(format_dict(v, indent + 2))
                    items.append(f"{' ' * indent}}},")
                else:
                    items.append(f"{' ' * indent}{k}: {v},")
            return items

        config_dict = self.to_dict()
        formatted_items = format_dict(config_dict)
        formatted_str = "\n".join(formatted_items)
        print(f"Config: {{\n{formatted_str}\n}}")
