"""
Module de gestion de l'entraînement du DQN.
Ce module contient la classe Trainer qui orchestre le processus d'entraînement,
gère les interactions avec l'environnement et met à jour l'agent.
"""

import os
from datetime import datetime

from common.utils.config import Config
from common.environnement.atarienv import AtariEnv
from common.preprocessing.preprocessor import Preprocessor

from algorithms.dqn.dqn_agent import DQNAgent
from algorithms.dqn.utils.logger import TrainingLogger
from algorithms.dqn.utils.evaluator import Evaluator

class Trainer:
    """
    Classe responsable de l'entraînement de l'agent DQN.
    
    Cette classe gère :
    - Les interactions avec l'environnement
    - La collecte d'expériences
    - La mise à jour de l'agent
    - La sauvegarde des checkpoints
    - L'évaluation périodique
    """
    
    def __init__(self, config: Config, agent: DQNAgent, environment: AtariEnv, preprocessor: Preprocessor):
        self.config = config
        
        # Création du dossier des sorties
        self.output_path = os.path.join(config.output_dir, config.name)
        if os.path.exists(self.output_path):
            raise FileExistsError(f"Le répertoire '{self.output_path}' existe déjà. Danger de remplacement des résultats !")
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Sorties enregistrées dans le répertoire: {self.output_path}")
        
        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor
        self.logger = TrainingLogger(config, self.output_path, self.agent.input_shape, self.agent.output_shape)
        self.evaluator = Evaluator(config, self.environment, self.preprocessor, self.logger)
            
        # Suivi pour l'évaluation périodique
        self.next_eval_checkpoint = self.config.training.eval_interval_steps
        self.next_target_update = self.config.dqn.hyperparameters.target_update
        
        self.log_every_n_episodes = self.config.training.log_every_n_episodes
        self.num_episode = 0
        self.total_steps = 0
        self.best_height = 1

    # Fonction principale d'entraînement, avec boucle jusqu'à atteindre le nombre de steps max
    def train(self):
        
        episode_best_height = 1
        print()
        
        # Boucle principale pour effectuer tous les steps d'entraînement
        while self.total_steps < self.config.training.max_training_steps:
            #affichage et log tout les n episodes
            if self.num_episode % self.log_every_n_episodes == 0:
                debut_episode = datetime.now()
                print(f"{debut_episode.strftime('%Y-%m-%d %H:%M:%S')} - Début Episode {self.num_episode:,} - Last bh/best bh: {episode_best_height:.2f}/{self.best_height:.2f} - Total steps: {self.total_steps:,}/{self.config.training.max_training_steps:,}")
                episode_frogger_score = 0
                episode_score = 0
                actions_list = []
                rewards_list = []
            
            # Réinitialisation de l'environnement
            frame, _ = self.environment.reset()

            # Prétraitement de la frame initiale
            # Retourne la frame prétraitée et la liste de features extraites.
            processed_frame, features = self.preprocessor.preprocess(frame)

            # Boucle pour stacker les frames initiales si nécessaire, uniquement pour le premier step (preprocessor ce charge du reste) 
            if self.config.preprocessing.frame_stacking > 1:
                for _ in range(self.config.preprocessing.frame_stacking - 1):
                    frame, _, _, _, _ = self.environment.step(0)  # Effectue une action "noop" (0)
                    processed_frame, features = self.preprocessor.preprocess(frame)

            done = False
            episode_best_height = 1
            episode_steps = 0
            
            while not done and episode_steps < self.config.env.max_episode_steps:
                
                #si le nombre de features est supérieur à 0, alors on envoie uniquement les features défini dans le config, sinon on envoie None
                features_selected = features[:self.config.dqn.network.num_features] if self.config.dqn.network.num_features > 0 else None
                
                # Sélection de l'action
                action = self.agent.select_action(processed_frame, features_selected)
                
                # Exécution de l'action
                next_frame, frogger_reward, terminated, truncated, info = self.environment.step(action)
                done = terminated or truncated

                # Prétraitement de la frame suivante
                next_processed_frame, next_features = self.preprocessor.preprocess(next_frame)
                
                # Sélection des features si nécessaire, sinon on envoie None pour les features supplémentaires (one-liner)
                next_features_selected = next_features[:self.config.dqn.network.num_features] if self.config.dqn.network.num_features > 0 else None
                
                # Calcul de la récompense obtenue par l'action
                reward = self.environment.get_reward(
                    action=action,
                    reward=frogger_reward,
                    prev_position=(features[0], features[1]),
                    new_position=(next_features[0], next_features[1]),
                    terminated=terminated, truncated=truncated, info=info
                    )
                                
                # Stockage de l'expérience
                self.agent.store_transition(processed_frame, action, reward, next_processed_frame, done, features_selected, next_features_selected)
                
                # Mise à jour de l'agent tout selon la fréquence définie dans le config
                if self.total_steps % self.config.dqn.hyperparameters.steps_frequency_update == 0:
                    self.agent.train_on_batch()
                
                # Mise à jour de l'itération
                processed_frame = next_processed_frame
                features = next_features
                
                # Mise à jour du suivi de l'épisode
                self.total_steps += 1
                episode_steps += 1
                self.agent.steps_count += 1
                episode_best_height = min(episode_best_height, features[1])
                
                if self.num_episode % self.log_every_n_episodes == 0:
                    episode_frogger_score += frogger_reward
                    episode_score += reward
                    actions_list.append(action)
                    rewards_list.append(reward)

            self.best_height = min(self.best_height, episode_best_height)
            
            # Mise à jour de l'agent après chaque épisode
            loss, td_error = self.agent.train_on_batch()

            # Mise à jour du target model selon la fréquence définie dans le config
            if self.total_steps >= self.next_target_update:
                self.agent.update_target_model()
                self.next_target_update += self.config.dqn.hyperparameters.target_update
            
            if self.num_episode % self.log_every_n_episodes == 0:
                # Logging des informations de l'épisode
                self.logger.log_episode(
                    debut_episode,
                    self.num_episode,
                    self.total_steps,
                    episode_steps,
                    self.agent.steps_count,
                    episode_frogger_score,
                    episode_score,
                    actions_list,
                    rewards_list,
                    episode_best_height,
                    loss,
                    td_error,
                    self.agent.epsilon,
                    self.agent.optimizer.param_groups[0]['lr'],
                    self.agent.replay_buffer.max_priority,
                    )
            
            self.num_episode += 1
                
            # Évaluation périodique de l'agent
            if self.total_steps >= self.next_eval_checkpoint:
                print(f"Évaluation de l'agent à {self.next_eval_checkpoint} steps ...")
                mean_reward = self.evaluator.evaluate(self.agent)
                print(f"Moyenne des récompenses d'évaluation sur {self.config.evaluate.num_parties} parties: {mean_reward:.2f}")
                # Prochain point de contrôle pour l'évaluation
                self.next_eval_checkpoint += self.config.training.eval_interval_steps
                
        mean_reward = self.evaluator.evaluate(self.agent)
        print(f"Moyenne des récompenses d'évaluation sur {self.config.evaluate.num_parties} parties: {mean_reward:.2f}")
    
    def save_checkpoint(self):
        """Sauvegarde un checkpoint de l'entraînement."""      
        path = os.path.join(
            self.output_path,
            f'checkpoint_episode_{self.num_episode}_steps_{self.agent.steps_count}.pt'
        )
        self.agent.save(path)
