from datetime import datetime
from pathlib import Path
import json
import re

import joblib
from nbformat import write
import torch
import torch.nn as nn
import random

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from base_model import ImageNetSubset
from submissions.my_team.model import ModelArchitecture


DATA_ROOT = Path("dataset")
LABELS_LIST = Path("dataset/labels.json")
OUTPUT = Path("weights.joblib")
OUTPUT_LOG = "logs/training_{}.log"

SEED = 67
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
FINAL_TEST_RATIO = 0.15

BATCH_SIZE = 32
EPOCHS = 2



def calculate_accuracy(model, data_loader):
    """
    Calculate the accuracy of the model on the provided data loader.

    Args:
        model: The trained model.
        data_loader: DataLoader for the dataset to evaluate.

    Returns:
        Accuracy as a percentage.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def train_one_epoch(epoch_index, tb_writer, model, optimizer, train_loader, val_loader, report_interval=10):
    """
    Train the model for one epoch.

    Args:
        epoch_index: Index of the current epoch.
        tb_writer: TensorBoard writer for logging.
        model: The model to train.
        optimizer: The optimizer for updating model parameters.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset. only used for calculating accuracy after the epoch.
    """
    model.train()
    running_loss = 0.0
    last_loss = 0.0
    for batch_index, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % report_interval == report_interval - 1:  # Log every `report_interval` batches
            last_loss = running_loss / report_interval
            timestamp = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
            print(f"{timestamp}: Epoch [{epoch_index + 1}], Batch [{batch_index + 1}], Loss: {last_loss:.4f}")
            tb_writer.add_scalar('training loss', last_loss, epoch_index * len(train_loader) + batch_index)
            running_loss = 0.0
    
    accuracy = calculate_accuracy(model, val_loader)
    print(f"Epoch [{epoch_index + 1}] completed. Training Accuracy: {accuracy:.2f}%")
    tb_writer.add_scalar('training accuracy', accuracy, epoch_index * len(train_loader)) # multiplied so the x-axis is consistent with the loss graph

    return last_loss

def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """
    model = ModelArchitecture()

    labels_list = json.load(open(LABELS_LIST)) # str -> str
    labels_list = {int(k): v for k, v in labels_list.items()} # int -> str

    # initialize seed
    torch.manual_seed(SEED)
    
    train_dataset = ImageNetSubset(DATA_ROOT,
                                   split=r"train_set\train",
                                   transform=IMAGE_TRANSFORMS_RANDOM_AUGMENTATIONS)
    validation_dataset = ImageNetSubset(DATA_ROOT,
                                        split=r"train_set\validation",
                                        transform=IMAGE_TRANSFORMS_RANDOM_AUGMENTATIONS)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(OUTPUT_LOG.format(timestamp))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # train the model
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Add training loop here
    for epoch in range(EPOCHS):
        last_loss = train_one_epoch(epoch, writer, model, optimizer, train_loader, val_loader)

    # evaluate the model on the test set
    accuracy = calculate_accuracy(model, val_loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # write to file
    writer.flush()
    writer.close()
    
    joblib.dump(model.state_dict(), "weights.joblib")
    print("Saved trained weights.joblib")

if __name__ == "__main__":
    main()