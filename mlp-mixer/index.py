import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.CIFAR10(
    root="dataset",
    download=False,
    transform=ToTensor()
)

valid_data = datasets.CIFAR10(
    root="dataset",
    download=False,
    transform=ToTensor()
)

from torch.utils.data import DataLoader
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
val_loader = DataLoader(training_data, batch_size=32, shuffle=True)

from MlpMixer import MLP_Mixer

network = MLP_Mixer(blocks=5, patches=8, DS=32, DC=32, 
    channels=64, height=32, width=32, channels_image=3, cls=10)

from torch import nn
from torch import optim
# from torch import cuda

device = "cpu"
# if cuda.is_available():
#     device = "cuda:0"

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)

def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = network(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 5 == 1:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(train_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

import time
# from torch.utils.tensorboard import SummaryWriter
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = str(time.time())
# writer = SummaryWriter('runs/cifair10_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    network.train(True)
    avg_loss = train_one_epoch()

    # We don't need gradients on to do reporting
    network.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(val_loader):
        vinputs, vlabels = vdata
        voutputs = network(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    # writer.add_scalars('Training vs. Validation Loss',
                    # { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    # epoch_number + 1)
    # writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(network.state_dict(), model_path)

    epoch_number += 1