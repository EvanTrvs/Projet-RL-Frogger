from common.environnement.atarienv import AtariEnv
from common.preprocessing.preprocessor import Preprocessor
from common.utils.config import Config

import numpy as np
import matplotlib.pyplot as plt

def afficher_image(image):
    # Affiche une image ou une série d'images à partir d'un tableau NumPy

    if image.ndim == 2:
        # Image en niveaux de gris
        plt.imshow(image, cmap='gray')
        plt.title('Image en niveaux de gris')
        plt.axis('off')
        plt.show()
    elif image.ndim == 3:
        if image.shape[2] == 3:
            # Image RGB
            plt.imshow(image)
            plt.title('Image RGB')
            plt.axis('off')
            plt.show()
        else:
            # Plusieurs canaux, afficher chaque canal côte à côte
            n_channels = image.shape[2]
            fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 5))
            for i in range(n_channels):
                axes[i].imshow(image[:, :, i], cmap='gray')
                axes[i].set_title(f'Canal {i+1}')
                axes[i].axis('off')
            plt.show()
    else:
        raise ValueError("Le tableau doit être de dimension 2 ou 3.")


def play(config_path: str = None):
    
    if config_path:
        config = Config.from_json(config_path)
    else:
        config = Config()
        
    config.env.render_mode = "human"
    
    environment = AtariEnv(config)
    
    preprocessor = Preprocessor(config)
    
    total_steps = 0

    # Boucle de jeu
    while total_steps < config.training.max_training_steps:
        
        # Initialise l'environnement
        observation, info = environment.reset()
        score = 0
        done = False
        steps = 0

        
        # Prétraitement de la frame initiale
        # La méthode `preprocess` du préprocesseur applique diverses transformations à la frame brute
        # Elle retourne la frame prétraitée et la liste de features extraites.
        processed_frame, features = preprocessor.preprocess(observation)
            
        # Boucle pour stacker les frames initiales si nécessaire, uniquement pour le premier step (preprocessor ce charge du reste) 
        if config.preprocessing.frame_stacking > 1:
            for _ in range(config.preprocessing.frame_stacking - 1):
                frame, _, _, _, _ = environment.step(0)  # Effectue une action "noop" (0)
                processed_frame, features = preprocessor.preprocess(frame)
                        
        # Boucle d'épisode                
        while not done and steps < config.env.max_episode_steps:        
            
            # Demande à l'utilisateur de saisir une action
            print("Choisir une action (0: Rien, 1: Haut, 2: Droite, 3: Gauche, 4: Bas, 5: afficher jeu, 6: afficher jeu traitée):")
            try:
                action = int(input("Action: "))  # Convertit l'entrée utilisateur en entier
                if action == 5:
                    # Afficher l'image du jeu (observation)
                    afficher_image(observation)
                    continue  # Redemande une action après avoir affiché l'image
                elif action == 6:
                    # Afficher l'image traitée (processed_frame)
                    afficher_image(processed_frame)
                    continue  # Redemande une action après avoir affiché l'image
                elif action not in range(environment.action_space.n):
                    raise ValueError("Action invalide.")
            except ValueError as e:
                print(f"Entrée invalide : {e}. Veuillez entrer un nombre entre 0 et {environment.action_space.n - 1}.")
                continue  # Redemande une action si l'entrée est invalide
            
            
            # Effectue l'action dans l'environnement
            observation, frogger_reward, terminated, truncated, info = environment.step(action)
            
            next_processed_frame, next_features = preprocessor.preprocess(observation)
            
            done = terminated or truncated
            
            # Calcul de la récompense obtenue par l'action
            reward = environment.get_reward(
                action=action,
                reward=frogger_reward,
                prev_position=(features[0], features[1]),
                new_position=(next_features[0], next_features[1]),
                terminated=terminated, truncated=truncated, info=info
                )

            # Met à jour le score
            score += reward
                   
            # Affiche le score actuel
            print(f"Action:{action}, Step: {steps}, Score: {score}, Reward: {reward}, hauteur: {next_features[1]}")
            
            steps += 1    
            total_steps += 1

            features = next_features   
            processed_frame = next_processed_frame


        print(f"\nEpisode terminé, score final: {score}")

    # Ferme l'environnement
    environment.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jouer comme un agent à l'environnement Atari Frogger")
    parser.add_argument("--config", help="Chemin vers le fichier de configuration", default=None)
    
    args = parser.parse_args()
    play(args.config) 
