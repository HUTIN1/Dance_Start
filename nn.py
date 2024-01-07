import torch.nn as nn
import torch.nn.functional as F
import monai

from Skeleton import Skeleton

class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=5,stride=2,padding=2), #64 -> 32
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=5,stride=2,padding=2), #32 -> 16
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=5,stride=2,padding=2), #16 -> 8
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=5,stride=2,padding=2), #8 -> 4
            nn.ReLU()
        )
        self.linear = nn.Linear(128*4*4,1)


    def forward(self, input):
        img = self.model(input)
        l = img.view(-1,128*4*4,1,1).squeeze(2,3)
        out = F.sigmoid(self.linear(l)).squeeze()

        return out
    

class Generator(nn.Module):
    def __init__(self, ngpu=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3,64,5,stride=2,padding=2), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64,128,5,stride=2,padding=2), #32 -> 16
            nn.ReLU(),
            nn.Conv2d(128,256,5,stride=2,padding=2), #16 -> 8
            nn.ReLU(),
            # nn.Dropout(p = 0.2),
            nn.ConvTranspose2d(256,128,4,stride=2,padding=1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,stride=2,padding=1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,stride=2,padding=1), # 32 -> 64
            nn.Tanh()
        )




    def forward(self, input):
        return self.model(input)
    

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, nchannel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nchannel,nchannel,5,padding=2)
        self.conv2 = nn.Conv2d(nchannel,nchannel,5,padding=2)
        self.relu = nn.ReLU()
        self.bathnorm = nn.BatchNorm2d(nchannel)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.bathnorm(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.bathnorm(x1)
        x = self.relu(x+x1)
        return x
    
class GeneratorResnet(nn.Module):
    def __init__(self, ngpu=0):
        super(GeneratorResnet, self).__init__()
        self.ngpu = ngpu
        self.model_down = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3,64,7,padding=0), # 64 -> 32
            nn.MaxPool2d((2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3,padding=1), #32 -> 16
            nn.MaxPool2d((2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,3,padding=1), #16 -> 8
            nn.MaxPool2d((2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
        )

        self.model_up = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,stride=2,padding=1), # 8 -> 16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,4,stride=2,padding=1), # 16 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,3,4,stride=2,padding=1), # 32 -> 64
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.ReflectionPad2d(3),
            nn.Conv2d(3,3,kernel_size=7,padding=0),
            nn.Tanh()
            )
        
        self.resnet = nn.Sequential(
            ResBlock(256),
            nn.Dropout(0.5),
            nn.BatchNorm2d(256),
            ResBlock(256),
            nn.BatchNorm2d(256),
            ResBlock(256),
            nn.Dropout(0.5),
            nn.BatchNorm2d(256),
            ResBlock(256),
            nn.BatchNorm2d(256),
            ResBlock(256),
            nn.Dropout(0.5),
            nn.BatchNorm2d(256),
        )




    def forward(self, input):
        input = self.model_down(input)
        # input = self.resnet(input)
        input = self.model_up(input)
        return input
    


class GeneratorUnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(GeneratorUnet,self).__init__()
        self.unet = monai.networks.nets.UNet(
            spatial_dims = 2,
            in_channels = 3,
            out_channels = 3,
            channels=(16, 32, 64, 128,),
            strides=(2, 2, 2,),
            num_res_units=2,
        )
        self.last_layer = nn.Conv2d(3,3,1,stride=1)
    
    def forward(self,input):
        input = self.unet(input)
        input = F.tanh(self.last_layer(input))
        return input


class GenNNSkeToImage(nn.Module):
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim,16*16*3),
            nn.ReLU(),
            nn.Linear(16*16*3,64*64*3),
            nn.Sigmoid(),
        )
        print(self.model)

    def forward(self, z):
        z = self.model(z)
        img = z.unsqueeze(-1).unsqueeze(-1).view(-1,3,64,64)
        return img