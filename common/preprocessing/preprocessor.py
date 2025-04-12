"""
Module de prétraitement des images pour l'apprentissage par renforcement sur Atari.
Fournit une interface unifiée et configurable pour le prétraitement des images.
"""

import torch
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from collections import deque
from ..utils.config import Config, PreprocessingConfig
import cv2
from concurrent.futures import ThreadPoolExecutor
from .transforms import (
BaseTransform,
ConvertirNuancesGris,
Resize,
ConvertirFlou,
RognerBords,
MappageValeurPixel,
CentrerSurGrenouille,
trouver_coordonnees_grenouille
)


class Preprocessor:
    """
    Classe principale pour le prétraitement des images.
    Gère un pipeline de transformations configurable et l'extraction de features.
    """
    
    def __init__(self, config: Config):
        """
        Initialise le prétraitement avec la configuration fournie.
        
        Args:
            config (Dict[str, Any]): Configuration du prétraitement
        """
        self.config: PreprocessingConfig = config.preprocessing
        self.transforms = self._create_transforms()
        self.nb_frames_frame_stacking = self.config.frame_stacking
        self.stacked_frames = deque(maxlen=config.preprocessing.frame_stacking)
        self.last_correct_position_grenouille = [0.5,0.5]  # On stocker la dernière position correcte de la grenouille

    def _create_transforms(self) -> List[BaseTransform]:
        """Crée la liste des transformations à appliquer."""
        transforms = []

        # Rognage des bords
        if self.config.rognage_bords:
            transforms.append(RognerBords())

        # Rognage autour de la grenouille
        if self.config.centrer_grenouille[0] != 0 and self.config.centrer_grenouille[1] != 0:
            taille_x = self.config.centrer_grenouille[0]
            taille_y = self.config.centrer_grenouille[1]
            valeur_remplissage = self.config.centrer_grenouille[2]
            transforms.append(CentrerSurGrenouille(taille_x,
                                                   taille_y,
                                                   valeur_remplissage))

        # Mappage valeurs pixels
        if self.config.mappage_pixels:
            transforms.append(MappageValeurPixel())

        # Conversion en niveaux de gris
        if self.config.grayscale:
            transforms.append(ConvertirNuancesGris())

        if self.config.flou != [0, 0]:
            transforms.append(ConvertirFlou(self.config.flou))

        if self.config.downscaling != [0, 0]:
            if not self.config.mappage_pixels:
                transforms.append(Resize(self.config.downscaling))
            else:
                transforms.append(Resize(self.config.downscaling, interpolation=False))
            
        return transforms
    
    def _apply_transforms(self, image: np.ndarray) -> np.ndarray:
        """Applique les transformations à une image."""
        for transform in self.transforms:
            image = transform(image)
        return image
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Applique le pipeline de prétraitement à une image.
        :param frame: Image à traiter
        :return: Tuple image traité et features extraites (si demandé dans config)
        """
        # Application des transformations

        features = self.extraire_features(frame)

        processed_frame = self._apply_transforms(frame)

        # Gestion du frame Stacking
        if self.nb_frames_frame_stacking > 1:
            self.stacked_frames.append(processed_frame)
            processed_frame = self.gerer_frame_stacking()

        return processed_frame, features

    def gerer_frame_stacking(self) -> np.ndarray:
        """
        Gère le frame stacking en fonction de la configuration.
        :return: Image stackée.
        """
        if self.config.frame_skip == "none":
            frames = list(self.stacked_frames)
            required_frames = self.config.frame_stacking
        elif self.config.frame_skip == "first-mid-last":
            frames = [self.stacked_frames[0], self.stacked_frames[len(self.stacked_frames) // 2], self.stacked_frames[-1]]
            required_frames = 3
        elif self.config.frame_skip == "first-last":
            frames = [self.stacked_frames[0], self.stacked_frames[-1]]
            required_frames = 2
        else:
            frames = list(self.stacked_frames)
            required_frames = self.config.frame_stacking

        # Ajouter une dimension supplémentaire si les frames sont en 2D
        frames = [np.expand_dims(frame, axis=-1) if frame.ndim == 2 else frame for frame in frames]

        # Combler les frames manquantes avec des zéros si nécessaire
        while len(frames) < required_frames:
            zero_frame = np.zeros_like(frames[0])
            frames.append(zero_frame)

        # Concaténer les frames sur la 3ème dimension
        return np.concatenate(frames, axis=2)

    def extraire_features(self, frame: np.ndarray) -> list[int]:
        """
        Extrait les caractéristiques de l'image.
        :param frame: Image d'entrée.
        :return: Liste des caractéristiques extraites.
        """
        features = []
        coordonnees_grenouille = trouver_coordonnees_grenouille(frame)
        if coordonnees_grenouille is None:
            features.extend([self.last_correct_position_grenouille[0], self.last_correct_position_grenouille[1]])
        else:
            x_relatif = coordonnees_grenouille[0] / frame.shape[1]
            y_relatif = coordonnees_grenouille[1] / frame.shape[0]
            features.append(x_relatif)  # Position X normalisée
            features.append(y_relatif)  # Position Y normalisée
            self.last_correct_position_grenouille = [x_relatif, y_relatif]

        return features
