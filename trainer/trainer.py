import torch
import torch.functional as F


def dice_loss(pred, target, smooth=1e-8):
    # flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
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


def training_loop(model, criterion, optimizer, n_epochs, dataloaders, device):

    def train_step(x, y):
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    for epoch in range(n_epochs):
        model.train()
        # epoch batch averages
        val_losses = []
        train_losses = []
        train_samples = 0
        val_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            train_step(x_batch, y_batch)

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                model.eval()
                yhat = model(x)
                loss = criterion(yhat, y)
                val_losses.append(loss.item() * y.size(0))
                val_samples += y.size(0)

        with torch.no_grad():
            # check performance on the entire train set at the end of the epoch
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                model.eval()
                yhat = model(x)
                loss = criterion(yhat, y)
                train_losses.append(loss.item() * y.size(0))
                train_samples += y.size(0)

        print(
            "Epoch {}:  Train Losses: {}\n".format(
                epoch, sum(train_losses) / train_samples
            )
        )
        print(
            "Epoch {}:  Val Losses:   {}\n".format(
                epoch, sum(val_losses) / val_samples
            )
        )

    print(model.state_dict())
