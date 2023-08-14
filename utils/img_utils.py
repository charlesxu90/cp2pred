import cv2
import os
import logging
import random
from joblib import Parallel, delayed


logger = logging.getLogger(__name__)


class ImageAugmentation:

    @staticmethod
    def rotating(img, num_rotations=60):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        
        rotations = list(range(0, 360, num_rotations))
        i = random.randint(0, len(rotations) - 1)
        M = cv2.getRotationMatrix2D(center, rotations[i], 1.0)
        # logger.info(f"rotating {rotations[i]} degrees, with center {M}")
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(255, 255, 255))   # randomly rotate i*60 degrees
        
        # logger.info(f"original shape: {img.shape}, rotated shape: {rotated.shape}")
        return rotated

    @staticmethod
    def flipping(img):
        i = random.randint(-1, 2)
        flipped = cv2.flip(img, i)  # 0, vertical; 1, horizontal; -1, both
        return flipped

    def __call__(self, img):
        img_aug = self.rotating(img)
        img_aug = self.flipping(img_aug)
        return img_aug
