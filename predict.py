# -*- coding: utf-8 -*-

import argparse
import logging
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from augment import AUGMENTATIONS_TEST
'''from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
'''

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    def process(full_img):
        img = Image.open(full_img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        preprocess = AUGMENTATIONS_TEST
        transformed = preprocess(image=img)
        img= transformed['image']
        img_trans = img.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        #print(mask.shape)
        return torch.from_numpy(img_trans).type(torch.FloatTensor).unsqueeze(0)
    img = process(full_img)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        net.train()
        mask = full_mask > out_threshold
        
    return mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def total_predict(ori_image,threshold=100):
    h_step = ori_image.shape[0] // 256
    w_step = ori_image.shape[1] // 256
    
    h_rest = -(ori_image.shape[0] - 256 * h_step)
    w_rest = -(ori_image.shape[1] - 256 * w_step)
    image_list = []
    predict_img = []
    for h in range(h_step):      # 截取片段
        for w in range(w_step):
            image_sample = ori_image[ (h*256):(h*256+256),
            (w*256 ) : (w*256 + 256), : ]
            image_list.append(image_sample)  
        image_list.append(ori_image[( h* 256) : (h*256 +256), -256:, :])
    for w in range(w_step-1):   
        image_list.append(ori_image[-256:, (w*256):(w*256 +256), :])
    image_list.append(ori_image[-256:, -256:, :])
    
    
    for image in image_list:        
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image=image.unsqueeze(0)
        pred1 = net(image)
        pred1 = pred1.squeeze(0).astype(np.int8)
        pred = pred>threshold
        predict_list.append(pred)
    # contact croped and predicted picture
    count_temp = 0
    tmp = np.ones([ori_image.shape[0]],ori_image.shape[1])
    for h in range(h_step):
        for w in range(w_step):
            tmp[
                h*256:(h+1)*256,
                w*256:(w+1)*256
            ] = predict_list[count_temp]
            count_temp += 1
        tmp[h *256 :(h+1) *256, w_rest:] = predict_list[count_temp][:, w_rest:]
        count_temp+=1
    for w in range(w_step -1):
        tmp[h_rest:, (w *256):(w*256+256)] = predict_list[count_temp][h_rest:,:]
        count_temp+=1
    tmp[-257:-1,-257:-1] = predict_list[count_temp][:, :]
    return tmp
if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))
        img = Image.open(fn)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
