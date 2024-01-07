from typing import Any
import torch
import os

class SaveNN:
    def __init__(self,name,path) -> None:
        self.name = name
        self.path = os.path.normpath(path)
        self.bestErr = 10000
        self.previous_save = None

    def __call__(self,nn, epoch, err) -> Any:
        if err < self.bestErr :
            self.bestErr = err
            if self.previous_save is not None:
                os.remove(self.previous_save)
            self.previous_save = os.path.join(self.path,f"{self.name}_{epoch}_{round(err.item(),4)}.pth")
            torch.save(nn,self.previous_save)