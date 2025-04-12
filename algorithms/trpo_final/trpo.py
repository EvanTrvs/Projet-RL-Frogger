import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import TRPO
import ale_py
from algorithms.trpo_final.wrapper_trpo.Wrapper import CustomPreprocessingWrapper
from algorithms.trpo_final.wrapper_trpo.Wrapper import CustomCallbackWrapper
from common.environnement.atarienv import AtariEnv
import os

# Création de l'environnement avec gymnasium
def trpo_final(env, config, prepro, shape):

    output_path = os.path.join(config.output_dir, config.name)
    if os.path.exists(output_path):
        raise FileExistsError(f"Le répertoire '{output_path}' existe déjà. Danger de remplacement des résultats !")
    os.makedirs(output_path, exist_ok=True)
    print(f"Sorties enregistrées dans le répertoire: {output_path}")


    new_env = CustomPreprocessingWrapper(env, prepro, shape)

    callback = CustomCallbackWrapper(log_filename=output_path)

    # Création et entraînement du modèle
    model = TRPO("CnnPolicy", new_env,verbose=1, learning_rate=0.001,
    gamma=config.trpo.hyperparameters.gamma,
    n_steps=2048,
    batch_size=config.trpo.hyperparameters.batch_size,
    gae_lambda=config.trpo.hyperparameters.tau,
    target_kl=config.trpo.hyperparameters.max_kl,
    cg_damping=config.trpo.hyperparameters.damping)
    model.learn(total_timesteps=config.training.max_training_steps, progress_bar=True, callback=callback)
    model.save("trpo_cartpole")
