"""
Module contenant les transformations d'images pour le prétraitement.
Chaque transformation est implémentée comme une classe avec une interface unifiée.
"""

from typing import Tuple, List, Any, Dict
import numpy as np
import cv2

class BaseTransform:
    """Classe de base pour toutes les transformations."""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Applique la transformation à l'image."""
        raise NotImplementedError
        
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calcule la forme de sortie après transformation."""
        return input_shape


class ConvertirNuancesGris(BaseTransform):
    """Conversion en niveaux de gris."""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Converti l'image rgb en nuances de gris
        :param image: image rgb à convertir
        :return: image en nuances de gris
        """
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # rgb -> bgr
            image_gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)  # Conversion en nuances de gris
            return image_gris
        else:
            return image


class Downscaling(BaseTransform):
    """Redimensionnement de l'image."""
    
    def __init__(self, ratio):
        """
        Applique un downscaling à l'image.
        :param ratio: le ratio de reduction à appliquer
        :return: image réduite
        """
        self.ratio = ratio
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un downscaling à l'image.
        :param image: image dont on va réduire les dimensions
        :return: image réduite
        """
        nouvelle_taille = (int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio))

        # l'interpolation INTER_AREA est plus adapté pour réduire la taille d'une image rapidement
        return cv2.resize(image, nouvelle_taille, interpolation=cv2.INTER_AREA)


class Resize(BaseTransform):
    """Redimensionnement de l'image."""

    def __init__(self, target_size: list[int], interpolation=True):
        """
        Applique un downscaling à l'image pour avoir une taille d'output specifique.
        :param target_size: Dimensions à atteindre
        """
        self.target_size = (target_size[1], target_size[0])  # (height, width) pour OpenCV
        self.interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applique un downscaling à l'image.
        :param image: image dont on va réduire les dimensions
        :return: image réduite
        """
        if self.interpolation:
            # Utilisation de l'interpolation pour une meilleure qualité
            return cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
        else:
            height_factor = image.shape[0] // self.target_size[0]
            width_factor = image.shape[1] // self.target_size[1]

            # Sous-échantillonnage simple
            reduced_image = image[::height_factor, ::width_factor]

            # Ajustement final si nécessaire
            return reduced_image[:self.target_size[0], :self.target_size[1]]


class ConvertirFlou(BaseTransform):
    def __init__(self, taille_noyau_gaussien: list[int]):
        """
        Utilise un flou gaussien pour appliquer un flou à l'image
        :param taille_noyau_gaussien: format (x,x) avec x impair, la taille du noyau gaussien
        """
        self.taille_noyau_gaussien = tuple(taille_noyau_gaussien)

    def __call__(self, image: np.ndarray):
        """
        Utilise un flou gaussien pour appliquer un flou à l'image
        :param image: image à flouter
        :param taille_noyau_gaussien: format (x,x) avec x impair, la taille du noyau gaussien
        :return: image floue
        """
        return cv2.GaussianBlur(image, self.taille_noyau_gaussien, 0)


class Normaliser(BaseTransform):

    def __call__(self, image: np.ndarray):
        """
        Nromalise l'image en uint8
        :param image:
        :return: image normalisée
        """
        return np.rint(image).astype(np.uint8)


class RognerBords(BaseTransform):
    def __call__(self, image: np.ndarray):
        """Enlève les bords de l'écran où la grenouille ne peut pas aller"""
        return image[12:-30, 8:-8, :]


class MappageValeurPixel(BaseTransform):
    def __init__(self, nouvelle_valeur_obstacle=150, nouvelle_valeur_safe=0, nouvelle_valeur_grenouille=100, nouvelle_valeur_eau=200):
        """
        Map chaque pixel à une valeur "obstacle" ou "safe"
        :param nouvelle_valeur_obstacle: La valeur à associer aux pixels obstacles comme les voitures
        :param nouvelle_valeur_safe: La valeur à associer aux pixels "safe" comme la route ou les troncs d'arbres
        """
        self.nouvelle_valeur_obstacle = nouvelle_valeur_obstacle
        self.nouvelle_valeur_safe = nouvelle_valeur_safe
        self.nouvelle_valeur_grenouille = nouvelle_valeur_grenouille
        self.nouvelle_valeur_eau = nouvelle_valeur_eau

        # Les valeurs rgb des pixels des eléments d'obstacles comme les voitures sur l'image de base
        self.valeurs_obstacles = [[195, 144, 61], [164,89,208], [82, 126, 45], [198, 89,179], [236,236,236]]
        self.valeur_grenouille=[110, 156, 66]
        self.valeur_eau=[[0,28,136]]

    def __call__(self, image: np.ndarray):
        """
        Map chaque pixel à une valeur "obstacle" ou "safe"
        :param image: image rgb non processed
        :return: nouvelle image avec les valeurs mappées
        """
        # On créé une nouvelle image avec la valeur "safe" par défaut
        nouvelle_image = np.full((image.shape[0], image.shape[1]), self.nouvelle_valeur_safe, dtype=np.uint8)

        # Création des masques pour chaque type de pixel, en ignorant les pixels en dessous de 25
        masques_obstacles = np.any(np.all(image[:, :, np.newaxis] == self.valeurs_obstacles, axis=-1), axis=-1)
        masque_grenouille = np.all(image[25:, :, :] == self.valeur_grenouille, axis=-1)
        masque_eau = np.all(image[:, :, :] == self.valeur_eau, axis=-1)

        # Application des masques
        nouvelle_image[masques_obstacles] = self.nouvelle_valeur_obstacle
        nouvelle_image[25:][masque_grenouille] = self.nouvelle_valeur_grenouille
        nouvelle_image[masque_eau] = self.nouvelle_valeur_eau

        return nouvelle_image


def trouver_coordonnees_grenouille(image_rgb_numpy):
    """
    Retourne les coordonnees du centre de la grenouille
    L'image doit être en rgb pour pouvoir trouver la grenouille
    :param image_rgb_numpy: l'image telle que retournée par l'environnement
    :return: les coordonnees du centre de la grenouille
    """
    # Couleur RGB de la grenouille
    couleur_grenouille = np.array([110, 156, 66])

    # On commence par chercher tous les pixels qui ont la couleur de la grenouille
    coordonnees = np.argwhere(np.all(image_rgb_numpy[25:, :, :] == couleur_grenouille, axis=-1))

    if len(coordonnees) == 0:

        # Si la grenouille a mangé une mouche elle devient jaune
        couleur_grenouille = np.array([160,160,52])
        coordonnees = np.argwhere(np.all(image_rgb_numpy[25:, :, :] == couleur_grenouille, axis=-1))

    # Si la grenouille est à l'écran, on trouve le centre de la grenouille
    if len(coordonnees) > 0:
        centre_coordonnees_y = int(np.mean(coordonnees[:, 0])) + 25
        centre_coordonnees_x = int(np.mean(coordonnees[:, 1]))

        #print(f"Centre de la grenouille : ({centre_coordonnees_y}, {centre_coordonnees_x})")
        return [centre_coordonnees_x, centre_coordonnees_y]

    else:
        return None


class CentrerSurGrenouille(BaseTransform):
    def __init__(self, taille_x, taille_y, valeur_remplissage=(0,0,0)):
        """
        Retourne une sous-partie de l'image avec la grenouille au centre. Si la grenouille n'est pas à l'écran, retourne
        une image remplie de valeur_remplissage.
        Si la grenouille est au bord de l'écran, les pixels en dehors de l'image sont remplacés par valeur_remplissage.
        :param taille_x: taille x de la nouvelle image à retourner
        :param taille_y: taille y de la nouvelle image à retourner
        :param valeur_remplissage: la valeur de remplissage
        :return: l'image rognée autour de la grenouille
        """
        self.taille_x = taille_x
        self.taille_y = taille_y
        self.valeur_remplissage = valeur_remplissage

    def __call__(self, image: np.ndarray):
        """
        Retourne une sous-partie de l'image avec la grenouille au centre. Si la grenouille n'est pas à l'écran, retourne
        une image remplie de valeur_remplissage.
        Si la grenouille est au bord de l'écran, les pixels en dehors de l'image sont remplacés par valeur_remplissage.
        :param image: image de base en rgb. Si c'est pas les valeurs rgb d'origine ça fonctionnera pas
        :return: l'image rognée autour de la grenouille
        """
        # On commence par créer une nouvelle image remplie avec la valeur de remplissage (pixels en dehors de l'écran)
        image_centree = np.full((self.taille_y, self.taille_x, 3), self.valeur_remplissage, dtype=np.uint8)
        coordonnees_grenouille = trouver_coordonnees_grenouille(image)

        if coordonnees_grenouille is not None:
            centre_coordonnees_x = coordonnees_grenouille[0]
            centre_coordonnees_y = coordonnees_grenouille[1]
            # On calcule les indices de début et de fin de la région autour de la grenouille
            y_start = max(0, centre_coordonnees_y - int(self.taille_y*0.7))
            y_end = min(image.shape[0], centre_coordonnees_y + int(self.taille_y * 0.3))
            x_start = max(0, centre_coordonnees_x - self.taille_x // 2)
            x_end = min(image.shape[1], centre_coordonnees_x + self.taille_x // 2)

            # On calcule les indices de début et de fin pour la copie dans la nouvelle image
            y_start_new = y_start - (centre_coordonnees_y - int(self.taille_y * 0.7))
            y_end_new = y_end_new = y_start_new + (y_end - y_start)
            x_start_new = x_start - (centre_coordonnees_x - self.taille_x // 2)
            x_end_new = x_start_new + (x_end - x_start)

            image_centree[y_start_new:y_end_new, x_start_new:x_end_new] = image[y_start:y_end, x_start:x_end]

        return image_centree


class FrameStacking(BaseTransform):
    def __init__(self, strategy='first_middle_last'):
        """
        Retourne une image ou plusieurs images résultants du procédé de frame stacking
        :param strategy: strategie à appliquer:
        - 'summed': la nouvelle image est la somme des images
        - 'averaged': la nouvelle image est la moyenne des images
        - 'first_middle_last': retourne la premiere image, celle du milieu et la derniere
        :return:
        """
        self.strategy=strategy

    def __call__(self, liste_images: np.ndarray):
        """
        Retourne une image ou plusieurs images résultant du procédé de frame stacking
        :param liste_images: les images à utiliser pour le frame stacking
        :return: Une image si strategy=summed ou averaged, 3 images si strategy==first_middle_last
        """
        if len(liste_images.shape) != 4:  # (frames, height, width, channels)
            return liste_images

        elif self.strategy == 'summed':
            return np.sum(liste_images, axis=0)
        elif self.strategy == 'averaged':
            return np.mean(liste_images, axis=0)
        elif self.strategy == 'first_middle_last':
            indice_middle = len(liste_images) // 2
            return [liste_images[0], liste_images[indice_middle], liste_images[-1]]




# Fonctions de preproccessing d'Evan:

class EdgeDetectionTransform(BaseTransform):
    """Détection de contours avec Canny."""
    
    def __init__(self, threshold1: float = 100, threshold2: float = 200):
        """
        Args:
            threshold1 (float): Premier seuil pour Canny
            threshold2 (float): Deuxième seuil pour Canny
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(image, self.threshold1, self.threshold2)


class HistogramEqualizationTransform(BaseTransform):
    """Égalisation d'histogramme."""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            # Égalisation CLAHE pour les images couleur
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Égalisation simple pour les images en niveaux de gris
            return cv2.equalizeHist(image) 