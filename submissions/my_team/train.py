from datetime import datetime
from pathlib import Path
import json
import re

import joblib
import torch
import torch.nn as nn

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

BATCH_SIZE = 128
EPOCHS = 5

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device) # Send to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0

def train_one_epoch(epoch_index, tb_writer, model, optimizer, criterion, train_loader, val_loader, device, report_interval=10):
    model.train()
    running_loss = 0.0
    last_loss = 0.0  # Safe fallback initialization
    
    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # Send to GPU
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % report_interval == report_interval - 1:
            last_loss = running_loss / report_interval
            timestamp = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
            print(f"{timestamp}: Epoch [{epoch_index + 1}], Batch [{batch_index + 1}], Loss: {last_loss:.4f}")
            tb_writer.add_scalar('training loss', last_loss, epoch_index * len(train_loader) + batch_index)
            running_loss = 0.0
    
    accuracy = calculate_accuracy(model, val_loader, device)
    print(f"Epoch [{epoch_index + 1}] completed. Validation Accuracy: {accuracy:.2f}%")
    tb_writer.add_scalar('validation accuracy', accuracy, epoch_index) # Log cleanly per epoch

    return last_loss


def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelArchitecture().to(device)

    labels_list = json.load(open(LABELS_LIST)) # str -> str
    labels_list = {int(k): v for k, v in labels_list.items()} # int -> str

    # initialize seed
    torch.manual_seed(SEED)

    
    train_dataset = ImageNetSubset(DATA_ROOT, r"train_set\train", transform=IMAGE_TRANSFORMS)
    validation_dataset = ImageNetSubset(DATA_ROOT, r"train_set\validation", transform=IMAGE_TRANSFORMS)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(OUTPUT_LOG.format(timestamp))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # train the model
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Add training loop here
    for epoch in range(EPOCHS):
        last_loss = train_one_epoch(epoch, writer, model, optimizer, criterion, train_loader, val_loader, device)

    # evaluate the model on the test set
    accuracy = calculate_accuracy(model, val_loader, device)
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # write to file
    writer.flush()
    writer.close()
    
    joblib.dump(model.state_dict(), "weights.joblib")
    print("Saved trained weights.joblib")


if __name__ == "__main__":
    main()