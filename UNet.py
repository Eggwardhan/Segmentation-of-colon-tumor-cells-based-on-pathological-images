# import _init_paths
import torch
import torch.nn as nn
#from layers import unetConv2, unetUp
#from utils import init_weights, count_param
from torch.autograd import Variable
import cv2 as cv
import torch.utils.data as Data
import os 
import glob
import random
from torchvision.datasets import ImageFolder
from torchvision import transforms
"""
class Loader(Data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, "/*.jpg"))
    
    def augment(self, image, flipCode):
        flip = cv.flip(image, flipCode)
        return flip
    
    def __getitem__(self, index):
        #根据index读取图片
        image_path = self.imgs_path[index]
        label_path = image_path.replace("image", "label")
        image = cv.imread(image_path)
        label = cv.imread(label_path)
        # bgr2gray
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        label = cv.cvtColor(label, cv.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label
    
    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
"""
class UNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_devonc=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        
        filters = [64, 128,256,512,1024]
        filters = [int(x / self.feature_scale) for x in filters]
        
         # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool(conv1)
        
        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64

        conv4 = self.conv4(maxpool3)         # 128*64*64
        maxpool4 = self.maxpool(conv4)       # 128*32*32

        center = self.center(maxpool4)       # 256*32*32
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up1 = self.up_concat1(up2,conv1)     # 16*512*512

        final = self.final(up1)

        return final

if __name__  == "__main__":
    print("#######test######")
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset= ImageFolder("/Users/eggwardhan/Documents/cv/DigestPath2019/Signet_ring_cell_dataset",
                         transform = data_transform)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                                                            batch_size=2, 
                                                                            shuffle=False)
    print("数据个数：", len(train_loader))


    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = Variable(torch.rand(2,1,64,64)).to(device)
    model = UNet().to(device)
    param = count_param(model)
    y = model(x)
    print("output shape:", y.shape)
    print("UNet total parameters: %.2fM(%d)" %(param/1e6,param) )
    """        