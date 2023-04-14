import torch
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import YoloPascalVocDataset
from loss import SumSquaredErrorLoss
from models import *


if __name__ == '__main__':      # Prevent recursive subprocess creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)         # Check for nan loss
    writer = SummaryWriter()
    now = datetime.now()

    model = YOLOv1ResNet().to(device)
    loss_function = SumSquaredErrorLoss()

    # Adam works better
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config.LEARNING_RATE,
    #     momentum=0.9,
    #     weight_decay=5E-4
    # )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    # Learning rate scheduler (NOT NEEDED)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=utils.scheduler_lambda
    # )

    # Load the dataset
    train_set = YoloPascalVocDataset('train', normalize=True, augment=True)
    test_set = YoloPascalVocDataset('test', normalize=True, augment=True)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True
    )

    # Create folders
    root = os.path.join(
        'models',
        'yolo_v1',
        now.strftime('%m_%d_%Y'),
        now.strftime('%H_%M_%S')
    )
    weight_dir = os.path.join(root, 'weights')
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)

    # Metrics
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    test_errors = np.empty((2, 0))


    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'test_losses'), test_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'test_errors'), test_errors)


    #####################
    #       Train       #
    #####################
    for epoch in tqdm(range(config.WARMUP_EPOCHS + config.EPOCHS), desc='Epoch'):
        model.train()
        train_loss = 0
        for data, labels, _ in tqdm(train_loader, desc='Train', leave=False):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            del data, labels

        # Step and graph scheduler once an epoch
        # writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        # scheduler.step()

        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
        writer.add_scalar('Loss/train', train_loss, epoch)

        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels, _ in tqdm(test_loader, desc='Test', leave=False):
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = model.forward(data)
                    loss = loss_function(predictions, labels)

                    test_loss += loss.item() / len(test_loader)
                    del data, labels
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.add_scalar('Loss/test', test_loss, epoch)
            save_metrics()
    save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
