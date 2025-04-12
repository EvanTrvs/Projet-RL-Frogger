"""
Module contenant les wrappers personnalisés pour l'environnement Atari.
Ces wrappers permettent de modifier le comportement de l'environnement
de manière modulaire et configurable.
"""
import gymnasium as gym
import numpy as np

class EpisodicLifeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 4
        self.live_reset = 4
        self.was_real_done = True

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.was_real_done = terminated
        
        if reward > 5 and not reward == 20:
            self.was_real_done = True
            terminated = True

        # Récupère le nombre de vies actuelles
        real_lives = info.get('lives', self.env.unwrapped.ale.lives())
        info['real_life'] = real_lives

        # Met à jour info['lives'] pour qu'il soit toujours 1, sauf en cas de perte de vie
        if real_lives < self.lives:
            terminated = True
            info['lives'] = 0
        else:
            info['lives'] = 1

        self.lives = real_lives
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        #print("live_reset", self.live_reset, "lives", self.lives,self.was_real_done)
        if self.was_real_done:
            observation, info = self.env.reset(**kwargs)
            #print("Hard reset trigger (terminated)")
        elif self.live_reset == self.lives:
            observation, info = self.env.reset(**kwargs)
            #print("Hard reset trigger")
        else:
            # No-op step to advance from terminal/lost life state without resetting
            observation, _, terminated, truncated, info = self.env.step(0)

            # Only call reset if the game is actually done
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
                #print("soft -> Hard reset trigger (terminated)")
            #else:
                #print("soft reset trigger")
                
        # Met à jour le nombre de vies initiales
        self.lives = info.get('lives', self.env.unwrapped.ale.lives())
        self.live_reset = self.lives
        info['real_life'] = self.lives
        info['lives'] = 1

        return observation, info

class NoopResetWrapper(gym.Wrapper):
    def __init__(self, env, noop_max=10):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            observation, _, terminated, truncated, info = self.env.step(self.noop_action)
            #print("Noop wrappers", _)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
        return observation, info
    
    
def reward_shaping_custom(action, reward, prev_position, new_position, terminated, truncated, info):
    if info.get('lost_life')==True:
        #Récompense pour avoir perdu une vie
        return -1
    elif reward==1:
        #Récompense pour avoir avance
        return 0.1
    elif reward >= 5 and not reward == 20:
        #Récompense pour avoir atteint le haut
        return 1
    elif new_position[1] > (prev_position[1]+0.01):
        #Récompense pour avoir recule
        return -0.05

    #Récompense pour le reste
    return 0