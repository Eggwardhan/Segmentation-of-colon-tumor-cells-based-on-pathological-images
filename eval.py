# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from tqdm import tqdm
from dice_loss import dice_coeff
import numpy as np
from sklearn.metrics import roc_auc_score


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0.0
    auc = 0.0
    trash=0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)/255

            with torch.no_grad():
                mask_pred = net(imgs)
                #print(mask_pred)
            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            elif net.n_classes==1:
                if hasattr(net,"deep_supervision"):
                    if net.deep_supervision==True:
                        for i in mask_pred:
                            auc_temp=0
                            pred = torch.sigmoid(i)
                            tot += dice_coeff(pred.to(device),true_masks.to(device),device)
                            pred=pred.cpu()
                            true_masks=true_masks.cpu()
                            try:
                                auc_temp+=roc_auc_score(pred.view(-1)>0.5,true_masks.view(-1))
                            except ValueError:
                                trash+=1
                        if trash!=mask_pred:
                            auc+=auc_temp/(len(mask_pred)-trash)

                        tot/=len(mask_pred)
                    else:
                        pred = torch.sigmoid(mask_pred)
                        pred = (pred > 0.5).float()
                         #print(pred)
                        tot += dice_coeff(pred.to(device), true_masks.to(device),device).item()
                        pred=pred.cpu()
                        true_masks=true_masks.cpu()
                        try:
                            auc+=roc_auc_score(pred.view(-1)>0.5,true_masks.view(-1))
                        except ValueError:
                            trash+=1
                else:
                        pred = torch.sigmoid(mask_pred)
                        pred = (pred > 0.5).float()
                         #print(pred)
                        tot += dice_coeff(pred.to(device), true_masks.to(device),device).item()
                        pred=pred.cpu()
                        true_masks=true_masks.cpu()
                        try:
                            auc+=roc_auc_score(pred.view(-1)>0.5,true_masks.view(-1))
                        except ValueError:
                            trash+=1
            pbar.update()

    net.train()
    return {"loss":tot / n_val,"auc":auc/n_val}

