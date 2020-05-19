import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from tqdm.auto import trange, tqdm

import time
from datetime import datetime
from collections import defaultdict


def dice_loss(pred, target, smooth = 1e-8):
     #flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()                            
    dice = (2.*intersection + smooth)/(pred.sum() + target.sum() + smooth)   
    return 1 - dice

def criterion(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * 0.5 + dice * (1 - 0.5)

    # metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)    

    return loss

def make_train_step(model, criterion, optimizer):
    """Builds function that performs a step in the train loop"""
    def train_step(x, y):

        model.train()
        yhat = model(x)

        loss = criterion(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    
    # Returns a function --> called within training loop
    return train_step


def training_loop(model, criterion, optimizer, n_epochs, dataloaders, device):
    
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_step = make_train_step(model, criterion, optimizer)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    for epoch in range(n_epochs):
        
        train_samples, val_samples = 0, 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            train_loss = train_step(x_batch, y_batch)
            train_losses.append(train_loss * y_batch.size(0)) # NO NEED FOR .data.cpu().numpy()? ****
            train_samples += y_batch.size(0)
            
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                model.eval()

                yhat = model(x_val)
                val_loss = criterion(y_val, yhat)
                val_losses.append(val_loss * y_batch.size(0)) # NO NEED FOR .data.cpu().numpy()? ****
                val_samples += y_batch.size(0)

        print("Epoch {}:  Train Losses: {}\n".format(epoch, np.sum(train_losses) / train_samples))
        print("Epoch {}:  Val Losses:   {}\n".format(epoch, np.sum(val_losses) / val_samples))

    print(model.state_dict())