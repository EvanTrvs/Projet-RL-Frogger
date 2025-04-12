"""
Module d'évaluation de l'agent DQN.
Ce module contient la classe Evaluator qui gère l'évaluation de l'agent,
la génération de GIFs et la sauvegarde des résultats.
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from algorithms.dqn.dqn_agent import DQNAgent
from common.environnement.atarienv import AtariEnv
from common.preprocessing.preprocessor import Preprocessor

from common.utils.config import Config
from algorithms.dqn.utils.logger import TrainingLogger
from common.utils.gif_maker import GifMaker

class Evaluator:
    """
    Classe responsable de l'évaluation de l'agent DQN.

    Cette classe gère :
    - L'évaluation de l'agent sur plusieurs épisodes
    - La génération de GIFs des épisodes
    - La sauvegarde des métriques d'évaluation
    - La visualisation des actions et des états
    """

    def __init__(self, config: Config, env: AtariEnv, preprocessor: Preprocessor, logger: TrainingLogger):
        """
        Initialise l'évaluateur.

        Args:
            config (Config): Configuration de l'évaluation
            logger (TrainingLogger): Logger pour enregistrer les métriques
        """
        self.config = config
        self.logger = logger
        self.output_path = self.logger.output_path
        self.gif_maker = GifMaker(fps=config.evaluate.gif_fps)
        self.gif_maker_preprocessed = GifMaker(fps=config.evaluate.gif_fps)
        self.env = env
        self.preprocessor = preprocessor

    def evaluate(self, agent: DQNAgent):
        """
        Évalue l'agent sur plusieurs épisodes.

        Args:
            agent (DQNAgent): Agent à évaluer
        """

        best_frogger_score = 0
        best_episode_frames = []
        best_episode_frames_preprocessed = []
        mean_reward = 0

        for episode in range(self.config.evaluate.num_parties):
            # Réinitialisation de l'environnement
            frame, _ = self.env.reset()
            state, features = self.preprocessor.preprocess(frame)
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_frogger_score = 0
            episode_best_height = 1
            actions_list = []
            rewards_list = []

            # Réinitialisation du GIF maker
            self.gif_maker.clear()
            self.gif_maker.add_frame(frame)
            self.gif_maker.add_frame(frame)

            self.gif_maker_preprocessed.clear()
            self.gif_maker_preprocessed.add_frame(state)
            self.gif_maker_preprocessed.add_frame(state)

            debut_episode = datetime.now()

            while not done and episode_steps < self.config.env.max_episode_steps:

                features_selected = features[:self.config.dqn.network.num_features] if self.config.dqn.network.num_features > 0 else None

                # Sélection de l'action (sans exploration)
                action = agent.select_action(state, features_selected, eval=False) #mettre features pour le contextual model

                # Exécution de l'action
                next_frame, frogger_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state, next_features = self.preprocessor.preprocess(next_frame)

                # Calcul de la récompense obtenue par l'action
                reward = self.env.get_reward(
                    action=action,
                    reward=frogger_reward,
                    prev_position=(features[0], features[1]),
                    new_position=(next_features[0], next_features[1]),
                    terminated=terminated, truncated=truncated, info=info
                )

                # Ajout de la frame au GIF
                self.gif_maker.add_frame(next_frame)
                self.gif_maker_preprocessed.add_frame(next_state)

                # Mise à jour des métriques
                episode_reward += reward
                episode_frogger_score += frogger_reward
                episode_steps += 1
                episode_best_height = min(episode_best_height, next_features[1])
                actions_list.append(action)
                rewards_list.append(reward)

                state = next_state
                features = next_features

            mean_reward += episode_reward

            # Log des informations de l'épisode
            if self.config.evaluate.trace_actions:
                self.logger.log_episode(
                    debut_episode,
                    episode,
                    agent.steps_count,
                    episode_steps,
                    agent.steps_count,
                    episode_frogger_score,
                    episode_reward,
                    actions_list,
                    rewards_list,
                    episode_best_height,
                    model_number=None,
                    EvalCsv=True
                )

            # Sauvegarde des frames de l'épisode avec le meilleur score Frogger
            if episode_frogger_score > best_frogger_score:
                best_frogger_score = episode_frogger_score
                best_episode_frames = self.gif_maker.frames.copy()
                best_episode_frames_preprocessed = self.gif_maker_preprocessed.frames.copy()

        self.gif_maker.add_frame(next_frame)
        self.gif_maker_preprocessed.add_frame(next_state)

        # Sauvegarde du GIF si nécessaire
        if self.config.evaluate.env_gif:
            self._save_episode_gif(best_episode_frames, agent.steps_count)

        # Sauvegarde du GIF des frames prétraitées si nécessaire
        if self.config.evaluate.observation_gif:
            self._save_episode_gif_preprocessed(best_episode_frames_preprocessed, agent.steps_count)

        return mean_reward / self.config.evaluate.num_parties

    def _save_episode_gif(self, frames: list, total_steps: int):
        """
        Sauvegarde le GIF de l'épisode.

        Args:
            frames (list): Liste des frames de l'épisode
            total_steps (int): Nombre total de steps de l'agent
        """
        gif_dir = os.path.join(self.output_path, 'gifs_eval')
        os.makedirs(gif_dir, exist_ok=True)
        path = os.path.join(gif_dir, f'eval_best_episode_{total_steps}_steps.gif')
        self.gif_maker.frames = frames
        self.gif_maker.save(path)

    def _save_episode_gif_preprocessed(self, frames: list, total_steps: int):
        """
        Sauvegarde le GIF des frames prétraitées de l'épisode.

        Args:
            frames (list): Liste des frames prétraitées de l'épisode
            total_steps (int): Nombre total de steps de l'agent
        """
        gif_dir = os.path.join(self.output_path, 'gifs_eval_prepro')
        os.makedirs(gif_dir, exist_ok=True)
        path = os.path.join(gif_dir, f'eval_best_episode_preprocessed_{total_steps}_steps.gif')
        self.gif_maker_preprocessed.frames = frames
        self.gif_maker_preprocessed.save(path)
