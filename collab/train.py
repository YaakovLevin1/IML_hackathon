from datetime import datetime
from pathlib import Path
import json
import os

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from base_model import ImageNetSubset
from model import ModelArchitecture

DATA_ROOT = Path("dataset")
LABELS_LIST = Path("dataset/labels.json")
OUTPUT = Path("weights.joblib")
OUTPUT_LOG = "logs/training_{}.log"

SEED = 67
TRAIN_RATIO = 0.8
TEST_RATIO = 0.15
FINAL_TEST_RATIO = 0.05

BATCH_SIZE = 64
EPOCHS = 10  # האילוץ שלנו

# תוספת AMP להאצת אימון על ה-GPU
scaler = GradScaler()


def train_one_epoch(epoch_index, tb_writer, model, optimizer, scheduler, train_loader, device, report_interval=10):
    model.train()
    running_loss = 0.0

    # Label Smoothing מסייע למניעת ביטחון יתר של הרשת
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # חישוב בחצי-דיוק לחיסכון בזיכרון וזמן
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # OneCycleLR מתעדכן אחרי כל Batch ולא כל Epoch!
        scheduler.step()

        running_loss += loss.item()
        if batch_index % report_interval == report_interval - 1:
            last_loss = running_loss / report_interval
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            print(
                f"{timestamp}: Epoch [{epoch_index + 1}/{EPOCHS}], Batch [{batch_index + 1}/{len(train_loader)}], Loss: {last_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            tb_writer.add_scalar('training loss', last_loss, epoch_index * len(train_loader) + batch_index)
            running_loss = 0.0

    return last_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = ModelArchitecture().to(device)

    os.makedirs("logs", exist_ok=True)

    with open(LABELS_LIST) as f:
        labels_list = json.load(f)
    labels_list = {int(k): v for k, v in labels_list.items()}

    torch.manual_seed(SEED)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(model.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(model.IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(model.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = ImageNetSubset(DATA_ROOT, "train", transform=train_transform)
    full_val_dataset = ImageNetSubset(DATA_ROOT, "train", transform=val_transform)

    dataset_size = len(full_train_dataset)
    indices = torch.randperm(dataset_size).tolist()

    train_size = int(TRAIN_RATIO * dataset_size)
    test_size = int(TEST_RATIO * dataset_size)

    train_dataset = Subset(full_train_dataset, indices[:train_size])
    validation_dataset = Subset(full_val_dataset, indices[train_size:train_size + test_size])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(OUTPUT_LOG.format(timestamp))

    # AdamW עם Weight Decay אגרסיבי לרגולריזציה
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    # נשק יום הדין ל-10 עידנים: OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,  # קצב למידה בשיא
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3  # מגיע לשיא אחרי 30% מהאימון
    )

    print("Starting aggressive 10-epoch training...")
    for epoch in range(EPOCHS):
        last_loss = train_one_epoch(epoch, writer, model, optimizer, scheduler, train_loader, device)

    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # בולידציה אנחנו לא צריכים חישוב בחצי-דיוק
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Validation Accuracy after 10 epochs: {accuracy:.2f}%")

    writer.flush()
    writer.close()

    joblib.dump(model.cpu().state_dict(), OUTPUT)
    print("Saved trained weights.joblib")


if __name__ == "__main__":
    main()