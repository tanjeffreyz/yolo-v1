import torch
import os
import config
import utils
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import YoloPascalVocDataset
from models import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)         # Check for nan loss
writer = SummaryWriter()
now = datetime.now()

model = YOLOv1ResNet().to(device)
loss_function = utils.SumSquaredErrorLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=utils.scheduler_lambda
)

# Load the dataset
transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_set = YoloPascalVocDataset('train', transform=transform)
test_set = YoloPascalVocDataset('test', transform=transform)

train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE)

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
    train_loss = 0
    for data, labels in tqdm(train_loader, desc='Train', leave=False):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model.forward(data)
        print('\n#############################')
        loss = loss_function(predictions, labels)
        print('TOTAL_LOSS', loss.item())
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader)
        del data, labels

    # Step and graph scheduler once an epoch
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    scheduler.step()

    train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
    writer.add_scalar('Loss/train', train_loss, epoch)

    if epoch % 4 == 0:
        with torch.no_grad():
            test_loss = 0
            for data, labels in tqdm(test_loader, desc='Test', leave=False):
                data = data.to(device)
                labels = labels.to(device)

                predictions = model.forward(data)
                print('\n#############################')
                loss = loss_function(predictions, labels)
                print('TOTAL_LOSS', loss.item())

                test_loss += loss.item() / len(test_loader)
                del data, labels
        test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
        writer.add_scalar('Loss/test', test_loss, epoch)
        save_metrics()
save_metrics()
torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
