"""
Module d'utilitaires pour la création de GIFs et autres outils pratiques.
"""

import numpy as np
import imageio.v3 as iio
from typing import List, Optional, Union, Tuple
import os

class GifMaker:
    """
    Classe utilitaire pour créer des GIFs à partir d'images numpy.
    Gère automatiquement la conversion des formats et la normalisation des images.
    """
    
    def __init__(self, fps: int = 10, normalize: bool = False):
        """
        Initialise le créateur de GIF.
        
        Args:
            fps (int): Images par seconde dans le GIF
            normalize (bool): Si True, normalise les images entre 0 et 255
        """
        self.fps = fps
        self.normalize = normalize
        self.frames: List[np.ndarray] = []
        
    def add_frame(self, frame: np.ndarray):
        """
        Ajoute une frame au GIF.
        
        Args:
            frame (np.ndarray): Image à ajouter (RGB, grayscale, ou autre format)
        """
        # Conversion en RGB si nécessaire
        if len(frame.shape) == 2:  # Grayscale
            frame = np.stack([frame] * 3, axis=-1)
        elif len(frame.shape) == 3 and frame.shape[-1] == 1:  # Grayscale avec canal
            frame = np.concatenate([frame] * 3, axis=-1)
            
        # Normalisation si activée
        if self.normalize:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.copy()
                
        self.frames.append(frame)
        
    def add_frames(self, frames: List[np.ndarray]):
        """
        Ajoute plusieurs frames au GIF.
        
        Args:
            frames (List[np.ndarray]): Liste d'images à ajouter
        """
        for frame in frames:
            self.add_frame(frame)
            
    def save(self, path: str):
        """
        Sauvegarde le GIF.
        
        Args:
            path (str): Chemin où sauvegarder le GIF
            duration (float, optional): Durée de chaque frame en secondes
        """
        if not self.frames:
            raise ValueError("Aucune frame n'a été ajoutée au GIF")
        
        # Sauvegarde du GIF
        iio.imwrite(path, self.frames, duration=(1/self.fps)*1000, loop=0)
        
        
    def clear(self):
        """Vide toutes les frames du GIF."""
        self.frames.clear()
        
    def __len__(self) -> int:
        """Retourne le nombre de frames dans le GIF."""
        return len(self.frames)
    
    @staticmethod
    def from_video(path: str, fps: int = 10) -> 'GifMaker':
        """
        Crée un GIF à partir d'une vidéo.
        
        Args:
            path (str): Chemin de la vidéo
            fps (int): Images par seconde dans le GIF
            
        Returns:
            GifMaker: Instance avec les frames de la vidéo
        """
        maker = GifMaker(fps=fps)
        reader = imageio.get_reader(path)
        for frame in reader:
            maker.add_frame(frame)
        return maker 