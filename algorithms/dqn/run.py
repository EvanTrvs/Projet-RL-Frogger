"""
Script principal pour l'entraînement du DQN.
Ce script orchestre l'ensemble du processus d'entraînement en utilisant
les différents composants (Trainer, Logger, Evaluator, etc.).
"""

import os
import sys

# Ajouter le répertoire parent au sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Optional

from common.utils.config import Config
from common.environnement.atarienv import AtariEnv
from common.preprocessing.preprocessor import Preprocessor

from algorithms.dqn.dqn_agent import DQNAgent
from algorithms.dqn.trainer import Trainer


def run(config_path: str, checkpoint_path: Optional[str] = None):
    """
    Fonction principale d'entraînement.
    
    Args:
        config_path (str): Chemin vers le fichier de configuration
        checkpoint_path (Optional[str]): Chemin vers un checkpoint à charger
    """
    # Chargement de la configuration json
    config = Config.from_json(config_path)
    config.display()

    # Initialiser le gestionnaire d'environnement avec la config json
    environment = AtariEnv(config)

    # Initialiser le gestionnaire de preprocessing avec la config json
    preprocessor = Preprocessor(config)

    # Récupération du shape post preprocessing
    frame, _  = environment.reset()
    preprocessed_frame, features = preprocessor.preprocess(frame)
    output_shape = environment.action_space.n
    print(f"Frame shape: {frame.shape}, Preprocessed Frame shape: {preprocessed_frame.shape}, Output shape: {output_shape}")
    print(f"Provided features shape: {len(features)}, Usefull features shape: {config.dqn.network.num_features}")
    if config.dqn.network.num_features > len(features):
        raise ValueError(f"Le nombre de features fournies ({features.shape[0]}) est inférieur au nombre de features nécessaires ({config.dqn.network.num_features})")

    device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    print(f"Device utilisé: {self.device}")

    if config.algorithm == "dqn":
        
        agent = DQNAgent(config, preprocessed_frame.shape, output_shape)
            
        # Chargement du checkpoint si fourni
        if checkpoint_path and os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            print(f"Checkpoint chargé: {checkpoint_path}")
        
        # Initialisation du trainer
        trainer = Trainer(config, agent, environment, preprocessor)
        
        try:
            # Lancement de l'entraînement
            trainer.train()
        except KeyboardInterrupt:
            print("\nEntraînement interrompu par l'utilisateur")
        finally:
            environment.close()
            
            # Sauvegarde finale
            trainer.save_checkpoint()
            print("Entraînement terminé")

    else:
        raise ValueError(f"Algorithme non supporté: {config.algorithm}, veuillez choisir entre 'dqn' et 'trpo'") 
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement d'un agent DQN")
    parser.add_argument("config", help="Chemin vers le fichier de configuration")
    parser.add_argument("--checkpoint", help="Chemin vers un checkpoint à charger")
    
    args = parser.parse_args()
    run(args.config, args.checkpoint) 
