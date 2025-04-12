import gymnasium as gym
import ale_py

from common.utils.config import Config
from common.environnement.wrappers import EpisodicLifeWrapper, NoopResetWrapper, reward_shaping_custom

"""
Module principal pour la gestion de l'environnement Atari.
Fournit une interface unifiée pour gérer l'environnement de jeu et ses wrappers.
"""

class AtariEnv(gym.Env):
    def __init__(self, config: Config):
        self.config = config

        # Crée l'environnement Atari
        self.env = gym.make(self.config.env.name, render_mode=self.config.env.render_mode, frameskip=2)

        # Applique les wrappers
        if self.config.env.EpisodicLifeEnv:
            self.env = EpisodicLifeWrapper(self.env)

        noop_max = 200 # pour avoir une frame de debut aleatoire
        self.env = NoopResetWrapper(self.env, noop_max=noop_max)
        self.skip_initial_frames = 220  # 400 frames à skip, 4 frames par action
        self.previous_lives = None

    def reset(self, **kwargs) :
        # Réinitialise l'environnement
        #print("reset atarienv")
        observation, info = self.env.reset()
            
        self.previous_lives = info.get('real_life', self.env.unwrapped.ale.lives())
        
        if not self.config.env.EpisodicLifeEnv or self.previous_lives == 4:
            self.skip_frames()
            
        info['lost_life'] = False
        return observation, info

    def skip_frames(self):
        # Passe rapidement les frames initiales sans les afficher
        for _ in range(self.skip_initial_frames):
            observation, reward, terminated, truncated, info = self.env.step(0)  # Action 0 pour ne rien faire
            #print("skip frames wrapper", _)
            if terminated or truncated:
                self.reset()

    def step(self, action=None):
        # Effectue l'action dans l'environnement
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation, _, new_terminated, new_truncated, info = self.env.step(0)
        
        terminated = new_terminated or terminated
        truncated = new_truncated or truncated
        
        if self.config.env.EpisodicLifeEnv:
            current_lives = info.get('real_life', self.env.unwrapped.ale.lives())
            info['lost_life'] = current_lives < self.previous_lives or (terminated and reward < 100)
            self.previous_lives = current_lives
        else:
            current_lives = info.get('lives', self.env.unwrapped.ale.lives())
            info['lost_life'] = current_lives < self.previous_lives or (terminated and reward < 100)
            self.previous_lives = current_lives
            
        if info['lost_life']:
            # Skip 15 frames to avoid the death animation
            for i in range(40):
                observation, _, _, _, _ = self.env.step(0)
                #print("step atarienv, skip death", i)

        return observation, reward, terminated, truncated, info

    def get_reward(self, action, reward, prev_position, new_position, terminated, truncated, info):
        
        if self.config.env.reward_shaping == "custom":
            return reward_shaping_custom(action, reward, prev_position, new_position, terminated, truncated, info)
        
        return reward

    def close(self):
        # Ferme l'environnement
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space