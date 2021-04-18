# -*- coding: utf-8 -*-

# version 1 mile
# refer : https://github.com/milesial/Pytorch-UNet/blob/master/train.py
import argparse
import logging
import os
import cv2
import sys
import cv2
import model.model as model
from eval import eval_net
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from PIL import Image
from tqdm import tqdm
import glob
from augment import AUGMENTATIONS_TRAIN,AUGMENTATIONS_TEST
Image.MAX_IMAGE_PIXELS=None

'''
from eval import eval_net
from unet import UNet
from utils.dataset import BasicDataset
'''
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, Dataset

dir_checkpoint= "./check_point/"

train_mask_dir = "../Colonoscopy_tissue_segment_dataset/train_mask" #create  train mask
train_dir = "../Colonoscopy_tissue_segment_dataset/train" # create train data
val_mask_dir = "../Colonoscopy_tissue_segment_dataset/val_mask"
val_dir = "../Colonoscopy_tissue_segment_dataset/val"


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, preprocess=None, scale=512, mask_suffix='_mask'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        #self.scale = scale
        self.preprocess= preprocess
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]   # get prefix or so-called id
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        #print("image_directory:{}  with {} files.\nmask_dir:{}".format(imgs_dir,len(self.ids),masks_dir))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def process(cls, pil_img,pil_mask,preprocess=None):
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    
        mask = np.array(pil_mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        if preprocess!= None:
            transformed = preprocess(image=img,mask=mask)
            img= transformed['image']
            mask= transformed['mask']
 
        elif self.preprocess!= None:
            transformed = self.preprocess(image=img,mask=mask)
            img= transformed['image']
            mask= transformed['mask']

        img_trans = img.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        #print(mask.shape)
        Grayimg = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask_trans = cv2.threshold(Grayimg, 12, 255,cv2.THRESH_BINARY)
        #cv2.imshow("sss",opencvImage)
        _ = {
            'image': torch.from_numpy(img_trans).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask_trans).type(torch.FloatTensor).unsqueeze(0)
        }

        return _

    def __getitem__(self, i):
        idx = self.ids[i]
        temp = os.path.join(self.masks_dir,idx+self.mask_suffix+'.*')
        mask_file = glob.glob(temp)

        img_file = glob.glob(os.path.join(self.imgs_dir , idx + '.*'))

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        #show_img(img)
        #img = self.process(img)
        #mask = self.process(mask)
        
        '''
        mask_trans = mask.transpose((2, 0, 1))
        if mask_trans.max() > 1:
            mask_trans = mask_trans / 255
        '''
        return self.process(img,mask)

def train_net(net,
              device,
              epochs=10,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              criterion="dice",
              img_scale=512):

    #dataset = BasicDataset(train_dir, train_mask_dir, img_scale)
    dataset = BasicDataset(train_dir, train_mask_dir,AUGMENTATIONS_TRAIN,img_scale)
    dataset2 = BasicDataset(val_dir,val_mask_dir,AUGMENTATIONS_TEST)
    n_train=len(dataset)
    n_val = len(dataset2)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    # 基本信息
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Criterion:       {criterion}
        Learning rate:   {lr}
        Training size:   {len(dataset)}
        Validation size: {len(dataset2)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    #优化器
#    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr,  weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.outc > 1 else 'max', patience=8)
    #if net.outc > 1:
    def dice_loss(input , target):
        input = torch.sigmoid(input)
        smooth = 0.00001
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1-((2.0*intersection + smooth)/(iflat.sum()+tflat.sum()+smooth))
    if  criterion=="bce":
        criterion = nn.BCEWithLogitsLoss()
    elif criterion ==  "cross":
        criterion == nn.CrossEntropyLoss()
    else:
        criterion = dice_loss

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                #mask_type = torch.float32 if net.outc == 1 else torch.long
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)
                #print("true mask shape: %s " % true_masks.shape)
                masks_pred = net(imgs)
                #print("predict mask shape:%s" % mask_pred.shape)
                loss = criterion(masks_pred,true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    logging.info('item_loss: {}'.format(epoch_loss))
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    logging.info('lr: {}'.format(optimizer.param_groups[0]['lr']))
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'{args.net}batch{args.batchsize}scale{args.scale}epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-n', '--net' ,dest="net",type=str,default="unet",
                        help="choose net like resnet")
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=256,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-c', '--criterion', dest='criterion',type=str,default='dice',
                        help = "criterion like bce crossentropyloss, dice_loss")

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net =  model.choose_net(args.net)
    net = net(in_channel=3,out_channel=1)
    logging.info(f'Network:\n'
                f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if args.load else "Transposed conv"} upscaling')

    if args.load:
       net.load_state_dict(
            torch.load(args.load, map_location=device)
            )
       logging.info(f'Model loaded from {args.load}')


    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  criterion=args.criterion,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), args.net+args.batch_size+'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

