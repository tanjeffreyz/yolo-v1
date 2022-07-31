import torch
import os
import config
import utils
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import YoloPascalVocDataset
from models import YOLOv1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
now = datetime.now()

model = YOLOv1().to(device)
loss_function = utils.SumSquaredErrorLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1E-3,
    momentum=0.9,
    weight_decay=0.005
)

# Learning rate schedulers
warmup_epochs = 5
increasing_sched = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=(warmup_epochs,),
    gamma=10
)
decreasing_sched = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=(warmup_epochs + 75, warmup_epochs + 105),
    gamma=0.1
)

# Load the dataset
train_set = YoloPascalVocDataset('train')
test_set = YoloPascalVocDataset('test')

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
for epoch in tqdm(range(warmup_epochs + config.EPOCHS), desc='Epoch'):
    train_loss = 0
    accuracy = 0
    for data, labels in tqdm(train_loader, desc='Train', leave=False):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model.forward(data)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader)
        accuracy += labels.eq(torch.argmax(predictions, 1)).sum().item() / len(train_set)
        del data, labels
    increasing_sched.step()       # Step schedulers once an epoch
    decreasing_sched.step()

    train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
    train_errors = np.append(train_errors, [[epoch], [1 - accuracy]], axis=1)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Error/train', 1 - accuracy, epoch)

    if epoch % 4 == 0:
        with torch.no_grad():
            test_loss = 0
            accuracy = 0
            for data, labels in tqdm(test_loader, desc='Test', leave=False):
                data = data.to(device)
                labels = labels.to(device)

                predictions = model.forward(data)
                loss = loss_function(predictions, labels)

                test_loss += loss.item() / len(test_loader)
                accuracy += labels.eq(torch.argmax(predictions, 1)).sum().item() / len(test_set)
                del data, labels
        test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
        test_errors = np.append(test_errors, [[epoch], [1 - accuracy]], axis=1)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Error/test', 1 - accuracy, epoch)

        save_metrics()
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(weight_dir, f'cp_{epoch}'))

save_metrics()
torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
