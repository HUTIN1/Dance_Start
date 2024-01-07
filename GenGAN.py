
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 
from nn import GeneratorResnet, Discriminator
from Callback import CallBack
from SaveNN import SaveNN






class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False,):
        self.netG = GeneratorResnet().cuda()
        self.netD = Discriminator().cuda()
        self.real_label = 1.
        self.fake_label = 0.
        self.callback = CallBack()
        self.saveGen = SaveNN("DanceGenGan","data")
        self.saveDis = SaveNN("DanceDisGan","data")
        self.filenameGen = 'model/DanceGenGAN.pth'
        self.filenameDis = 'model/DanceDisGan.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            
                            ])
        source_transform = transforms.Compose([
                        SkeToImageTransform(64)
                        
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform,source_transform=source_transform)
        print(f'len dataset {len(self.dataset)}')
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile :#and os.path.isfile(self.filenameGen):
            print("GenGAN: Load=", self.filenameGen, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(os.path.join(os.getcwd(),os.path.normpath("model\DanceGenGan_10_13.2292.pth")))
            self.netD = torch.load(os.path.join(os.getcwd(),os.path.normpath("model\DanceDisGan_9_0.0135.pth")))


    def train(self, n_epochs=20):
        criterion = nn.BCELoss()
        optimiezrD = torch.optim.Adam(self.netD.parameters(),lr=0.001)
        optimiezrG = torch.optim.Adam(self.netG.parameters(),lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.netD.eval()
        
        for epoch in tqdm(range(n_epochs)):
            ErrG = 0
            ErrD = 0
            for idx , (img_ske,im) in enumerate(self.dataloader):

               
                self.netD.zero_grad()
                img_ske = img_ske.to(device).to(torch.float32)
                im = im.to(device)
                
                #real image
                output_real = self.netD(im).squeeze()
                # print(f'min {torch.min(output_real)}')
                
                batch = im.size(0)
                label = torch.full((batch,), self.real_label, dtype=torch.float, device=device)
                
                errD_real = criterion(output_real,label)
                errD_real.backward()
                # D_x = output_real.mean().item()
                # print('first backward')



                #fake image
                
                fake = self.netG(img_ske)
                label.fill_(self.fake_label)
                
                output = self.netD(fake.detach()).view(-1)
                # print(f'min {torch.min(output_real)}')
                errD_fake = criterion(output,label)

                errD_fake.backward()
                # print('second backward')
                
                errD = errD_real + errD_fake
                ErrD+=errD
                
                optimiezrD.step()
                
                
                self.netG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake).view(-1)
                # print(f'min {torch.min(output)}')
                errG = criterion(output,label)
                ErrG += errG

                errG.backward()
                optimiezrG.step()

            print(f'loss Generator {ErrG/idx}')
            print(f'loss Descrimator {ErrD/idx}')
            if epoch % 2 == 0 :
                self.callback(fake,im, epoch)
            # cv2.imshow('Image', fake[0].detach().permute(1,2,0).cpu().numpy())

            # self.saveDis(self.netD,epoch,ErrD/idx)
            # self.saveGen(self.netG,epoch,ErrG/idx)

        torch.save(self.netG, self.filenameGen)
        torch.save(self.netD, self.filenameDis)


    def generate(self, ske):           # TP-TODO
        """ generator of image from skeleton """
        # ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        # ske_t = ske_t.to(torch.float32)
        # ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        image = white_image = np.ones((64, 64, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).unsqueeze(0).to(self.device).to(torch.float)
        # print(image.shape)
        normalized_output = self.netG(image)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = os.path.normpath("data/taichi1.mp4")
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    #if False:
    if False:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, True)
        gen.train(5) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file   


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

