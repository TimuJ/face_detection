import json
import torch

from tqdm import tqdm

import numpy as np

import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from src.dataset import FaceLandmarksDataset
from src.model import FaceKeypointResNet34

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fit(model, dataloader, dataset, optimizer, criterion, batch_size):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(dataset)/batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        if torch.cuda.is_available():
            image, keypoints = data['image'].to(
                device), data['keypoints'].to(device)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss/counter
    return train_loss

# validation function


def validate(model, dataloader, data, optimizer, criterion, epoch, batch_size):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data['image'].to(
                device), data['keypoints'].to(device)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()

    valid_loss = valid_running_loss/counter
    return valid_loss


def main():
    with open('train_config.json', 'r') as f:
        config = json.load(f)

    path_to_train = str(config['path_to_train'])
    batch_size = int(config['batch_size'])
    path_to_val = str(config['path_to_val'])
    lr = float(config['learning_rate'])
    epochs = int(config['epochs'])
    path_to_weights = str(config['weights_directory'])
    path_to_model = str(config['model_directory'])

    # change channels to [C, H, W] and put to Tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # making train and validation DataLoaders
    trainset = FaceLandmarksDataset(
        dir_to_jpgs=path_to_train, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    valset = FaceLandmarksDataset(
        dir_to_jpgs=path_to_val, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True)

    # model
    model = FaceKeypointResNet34(requires_grad=True).to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # MSELoss
    criterion = nn.MSELoss()

    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = fit(model, train_loader,
                               trainset, optimizer, criterion, batch_size)
        val_epoch_loss = validate(
            model, val_loader, valset, optimizer, criterion, epoch, batch_size)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {val_epoch_loss:.4f}')

    torch.save(model.state_dict(), path_to_weights)

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, path_to_model)

    print(
        f'Training finished, weights were saved to {path_to_weights}, model was saved to {path_to_model}')


if __name__ == '__main__':
    main()
