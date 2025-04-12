from stable_baselines3.common.callbacks import BaseCallback
import common.preprocessing.preprocessor as preprocessor
import gymnasium as gym
from common.utils.config import Config
import numpy as np
import csv
import time

class CustomPreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env, prepro, shape):
        super().__init__(env)
        self.preprocessor = prepro
        new_shape = (shape[0], shape[1], 1) if len(shape) == 2 else shape
        print("Shape new : ", new_shape)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)
        self.last_feature_1 = 1

    def observation(self, obs):
        obs, features = self.preprocessor.preprocess(obs)
        if features[1] < self.last_feature_1:
            self.last_feature_1 = features[1]
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=-1)
        return obs.astype(np.uint8)

    def reset_last_feature_1(self):
        self.last_feature_1 = 1

class CustomCallbackWrapper(BaseCallback):
    def __init__(self, log_filename, verbose=1):
        super().__init__(verbose)
        self.log_filename = log_filename + "result.csv"
        self.start_time = time.time()
        self.last_logged_time = time.time()
        self.episode_rewards = []
        self.current_reward = 0
        self.episode_len = []
        self.current_len = 0
        self.episode_height = []

        # Création du fichier CSV avec en-têtes
        with open(self.log_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestep", "ep_len_mean", "ep_rew_mean",
                "time_elapsed",  "explained_variance",
                "is_line_search_success", "kl_divergence_loss",
                "policy_objective", "value_loss",
                "height", "list_height", "list_reward", "list_len"
            ])

    def _get_custom_env(self):
        env = self.training_env
        if hasattr(env, "envs"):
            env = env.envs[0]  # DummyVecEnv
        if hasattr(env, "env"):
            env = env.env  # Monitor
        return env  # Devrait être CustomPreprocessingWrapper


    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]

        # Cumule le reward (on suppose ici 1 env seulement)
        self.current_reward += rewards[0]
        self.current_len += 1

        # Vérifie si l’épisode est terminé
        dones = self.locals["dones"]
        if dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_len.append(self.current_len)
            self.current_reward = 0
            self.current_len = 0

            custom_env = self._get_custom_env()
            height = getattr(custom_env, "last_feature_1", "N/A")
            self.episode_height.append(height)
            if hasattr(custom_env, "reset_last_feature_1"):
                custom_env.reset_last_feature_1()

        return True

    def _on_rollout_end(self) -> bool:
        log_data = self.logger.name_to_value

        now = time.time()
        total_elapsed_real = now - self.start_time
        step_elapsed_real = now - self.last_logged_time
        self.last_logged_time = now

        # Récupération des métriques si elles existent
        timestep = self.num_timesteps
        time_elapsed = step_elapsed_real
        explained_variance = log_data.get("train/explained_variance", "N/A")
        is_line_search_success = log_data.get("train/is_line_search_success", "N/A")
        kl_divergence_loss = log_data.get("train/kl_divergence_loss", "N/A")
        policy_objective = log_data.get("train/policy_objective", "N/A")
        value_loss = log_data.get("train/value_loss", "N/A")
        ep_rew_mean = np.mean(self.episode_rewards)
        ep_len_mean = np.mean(self.episode_len)
        height = np.mean(self.episode_height)
        list_height = self.episode_height
        list_reward = self.episode_rewards
        list_len = self.episode_len

        # Reboot
        self.episode_rewards = []
        self.episode_len = []
        self.episode_height = []

        # Sauvegarde dans le fichier CSV
        with open(self.log_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestep, ep_len_mean, ep_rew_mean,
                time_elapsed, explained_variance,
                is_line_search_success, kl_divergence_loss,
                policy_objective, value_loss, height,
                list_height, list_reward, list_len
            ])

        return True