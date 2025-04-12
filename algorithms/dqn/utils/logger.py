"""
Module de logging pour l'entraînement du DQN.
Ce module contient la classe TrainingLogger qui gère l'enregistrement
des métriques d'entraînement et la sauvegarde des logs.
"""

import os
import csv
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from common.utils.config import Config

class TrainingLogger:
    """
    Classe responsable du logging pendant l'entraînement.

    Cette classe gère :
    - L'enregistrement des métriques d'entraînement
    - La sauvegarde des logs dans des fichiers
    - Le suivi des épisodes et des steps
    - La gestion des checkpoints
    """

    def __init__(self, config: Config, output_path, input_shape, output_shape):
        self.config = config
        self.output_path = output_path
        self.training_name = self.config.name
        
        #Sauvegarder la configuration
        self.config.save(self.output_path)

        # Chemin du fichier CSV
        self.csv_path = os.path.join(self.output_path, f'{self.training_name}_training_metrics.csv')
        
        self.columns = [
            'datetime', 'total_steps', 'agent_steps', 'num_episode', 'episode_steps',
            'episode_duration', 'time_per_step', 'episode_frogger_score', 'episode_score',
            'episode_best_height', 'loss', 'td_error', 'epsilon', 'learning_rate', 'max_priority', 'actions_list', 'rewards_list', 'model_number'
        ]

        # Écriture des en-têtes si le fichier n'existe pas
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)
                
        if self.config.evaluate.trace_actions:
            self.csv_eval_path = os.path.join(self.output_path, f'{self.training_name}_evaluation_metrics.csv')
            
            if not os.path.isfile(self.csv_eval_path):
                with open(self.csv_eval_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.columns)
                
        #self.log_path = os.path.join(self.output_path, f'{self.training_name}_training.log')
        #self._write_log_introduction()

    def _write_log_introduction(self):
        """Écrit une introduction dans le fichier de log avec les informations clés sur l'entraînement."""
        introduction = (
            f"Training Name: {self.training_name}\n"
            f"Algorithm: {self.config.algorithm}\n"
            f"Input Shape: {self.input_shape}\t Output Shape: {self.output_shape}\n"
            f"Environment: {self.config.env['name']}\t EpisodicLifeEnv: {self.config.env['EpisodicLifeEnv']}\n"
            f"Max episode steps: {self.config.env['max_episode_steps']}\t Reward Shaping: {self.config.env['reward_shaping']}\n"
            f"Max Training Steps: {self.config.training['max_training_steps']: }\n"
            f"Evaluation Interval Steps: {self.config.training['eval_interval_steps']: }\n"
            f"Log Every N Episodes: {self.config.training['log_every_n_episodes']: }\n"
            f"==================================================================================================================="
            "\n"
        )

        with open(self.log_path, mode='w') as file:
            file.write(introduction)
            

    def log_episode(self,
                    debut_episode: datetime,
                    num_episode: int,
                    total_steps: int,
                    episode_steps: int,
                    agent_steps: int,
                    episode_frogger_score: float,
                    episode_score: float,
                    actions_list: list,
                    rewards_list: list,
                    episode_best_height: float,
                    loss: float = None,
                    td_error: float = None,
                    epsilon: float = None,
                    learning_rate: float = None,
                    max_priority: float = None,
                    model_number: list = None,
                    EvalCsv: bool = False):
        """
        Enregistre les métriques d'un épisode dans un fichier CSV.

        Args:
            debut_episode (datetime): Heure de début de l'épisode.
            num_episode (int): Numéro de l'épisode.
            total_steps (int): Nombre total de steps.
            episode_steps (int): Nombre de steps dans l'épisode.
            agent_steps (int): Nombre de steps de l'agent.
            episode_frogger_score (float): Score Frogger de l'épisode.
            episode_score (float): Score de l'épisode.
            actions_list (list): Liste des actions.
            rewards_list (list): Liste des récompenses.
            episode_best_height (float): Meilleure hauteur de l'épisode.
            loss (float): Perte de l'épisode.
            td_error (float): Erreur TD de l'épisode.
        """
        datetime_actuel = datetime.now()
        episode_duration = (datetime_actuel - debut_episode).total_seconds()
        time_per_step = episode_duration / episode_steps if episode_steps > 0 else 0

        episode_data = [
            datetime_actuel.isoformat(),
            total_steps,
            agent_steps,
            num_episode,
            episode_steps,
            round(episode_duration, 3),
            round(time_per_step, 6),
            episode_frogger_score,
            episode_score,
            episode_best_height,
            loss,
            td_error,
            epsilon,
            learning_rate,
            max_priority,
            actions_list,
            rewards_list,
            model_number
        ]

        if EvalCsv:
            with open(self.csv_eval_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(episode_data)
        else:
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(episode_data)
            
        # Log des informations génériques
        #self._log_generic_info(episode_data)
    
    def _log_generic_info(self):
        """Écrit des informations génériques dans un fichier LOG."""
        with open(self.log_path, mode='a') as file:
            file.write(f"{datetime.now().isoformat()} - Episode {self.current_episode} - Reward: {self.metrics['episode_rewards'][-1]:.2f} - Length: {self.metrics['episode_lengths'][-1]}\n")