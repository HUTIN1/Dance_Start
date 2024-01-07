from typing import Any
import torch
import cv2
import os

class CallBack:
    def __init__(self) -> None:
        self.path = os.path.normpath("C:\Ecole\Master_ID3D\AI\DeepLearning\dance_start\Callback_image")

    def __call__(self, pred,target,epoch) -> Any:
        pred = pred.detach()[3,...].permute(1,2,0).cpu().numpy()
        target = target.detach()[3,...].permute(1,2,0).cpu().numpy()

        cv2.imwrite(os.path.join(self.path,f"pred_{epoch}.png"), (pred+1)/2*255)
        cv2.imwrite(os.path.join(self.path,f"target_{epoch}.png"), (target+1)/2*255)