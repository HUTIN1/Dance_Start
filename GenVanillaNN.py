import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import monai
from tqdm import tqdm

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from nn import Generator, GeneratorUnet,GeneratorResnet
from Callback import CallBack


torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):

        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image



class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )



    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)

        return ske, image

    
    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.cpu().detach().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output
    
    def tensor2image_2(self,normalized_image):
        image = normalized_image.detach().permute(1,2,0).cpu().numpy()
        return (image+1)/2*255



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)












class GenVanillaNN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        self.image_size = 128
        self.ske_to_image = SkeToImageTransform(self.image_size )
        self.callback = CallBack()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.netG = Generator().cuda()
        self.netG = GeneratorResnet().cuda()
        source_transform = transforms.Compose([
                        SkeToImageTransform(self.image_size)
                        
        ])
        self.loss = nn.MSELoss()
        self.filename = 'model/DanceGenVanillafromim.pth'
        self.optimizer = torch.optim.Adam(self.netG.parameters(),lr=0.001)

        tgt_transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.CenterCrop(self.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # [transforms.Resize((64, 64)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=source_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TP-TODO
        for epoch in tqdm(range(n_epochs)):
            Error = 0
            for idx , (im_skel,im) in enumerate(self.dataloader):
            # for idx , (skel,im) in enumerate(self.dataloader):
                # skel = skel.to(device).squeeze()
                im_skel = im_skel.to(self.device).to(torch.float)

                im = im.to(device)
                pred = self.netG(im_skel)
                error = self.loss(im,pred)
                error.backward()
                self.optimizer.step()
                Error += error.detach().item()
            
            if epoch % 10 == 0 :
                self.callback(pred,im,epoch)

            print(f'error {Error/idx}')


            torch.save(self.netG, self.filename)


    def generate(self, ske):
        """ generator of image from skeleton """
        # TP-TODO
        # ske_t = self.dataset.preprocessSkeleton(ske).squeeze()
        # ske_t_batch = ske_t.unsqueeze(0).cuda()       # make a batch
        # print(ske_t_batch.shape)
        # image = white_image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
        # ske.draw(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.transpose((2,0,1))
        # image = torch.from_numpy(image).unsqueeze(0).to(self.device).to(torch.float)
        image = torch.from_numpy(self.ske_to_image(ske)).unsqueeze(0).to(self.device).to(torch.float)
        normalized_output = self.netG(image)
        res = self.dataset.tensor2image_2(normalized_output[0,...])       # get image 0 from the batch
        return res




if __name__ == '__main__':
    force = False
    optSkeOrImage = 2           # use as input a skeleton (1) or an image with a skeleton drawed (2)
    n_epoch = 200  # 200
    train = 1 #False
    train = False

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)    # load from file        


    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
