
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        """
        videoSkeTgt : VideoSkeleton
        """
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ 
        ske : Skeleton
        generator of image from skeleton """

        # TP-TODO
        # empty = np.ones((64,64, 3), dtype=np.uint8)
        best_error = 1000000
        best_idx = -1
        for i in range(self.videoSkeletonTarget.skeCount()):
            error = self.videoSkeletonTarget.ske[i].distance(ske)
            if error < best_error :
                best_error = error
                best_idx = i
        # self.videoSkeletonTarget.ske[best_idx].draw(empty)
        return self.videoSkeletonTarget.readImage(best_idx)




