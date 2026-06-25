from datetime import datetime
from pathlib import Path
import json
import os
import numpy as np

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
BATCH_SIZE = 64  # הורדנו ל-64 כדי לתת למודל יותר צעדי למידה
EPOCHS = 50

scaler = GradScaler()

def cutmix_data(x, y, alpha=1.0):
    """פונקציה המבצעת CutMix על הנתונים"""
    indices = torch.randperm(x.size(0)).to(x.device)
    shuffled_x = x[indices]
    shuffled_y = y[indices]
    lam = np.random.beta(alpha, alpha)

    # חישוב אזור התיבה
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = shuffled_x[:, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y, shuffled_y, lam


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def train_one_epoch(epoch_index, tb_writer, model, optimizer, scheduler, train_loader, device, report_interval=10):
    model.train()
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # החלת CutMix ב-50% מהזמן
        if np.random.rand() < 0.5:
            images, targets_a, targets_b, lam = cutmix_data(images, labels)
            with autocast():
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        running_loss += loss.item()
        if batch_index % report_interval == report_interval - 1:
            last_loss = running_loss / report_interval
            print(
                f"Epoch [{epoch_index + 1}], Batch [{batch_index + 1}], Loss: {last_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            running_loss = 0.0
    return last_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelArchitecture().to(device)

    with open(LABELS_LIST) as f:
        labels_list = json.load(f)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(model.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(model.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_train = ImageNetSubset(DATA_ROOT, "train", transform=train_transform)
    full_val = ImageNetSubset(DATA_ROOT, "train", transform=val_transform)

    indices = torch.randperm(len(full_train)).tolist()
    train_dataset = Subset(full_train, indices[:int(TRAIN_RATIO * len(full_train))])
    val_dataset = Subset(full_val, indices[int(TRAIN_RATIO * len(full_train)):])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, steps_per_epoch=len(
        DataLoader(train_dataset, batch_size=BATCH_SIZE)), epochs=EPOCHS)

    print("Starting training with CutMix optimization...")
    for epoch in range(EPOCHS):
        train_one_epoch(epoch, None, model, optimizer, scheduler,
                        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2), device)

    print("Evaluating...")
    model.eval()
    # כאן יבוא לוגיקת הולידציה הרגילה שלך
    joblib.dump(model.cpu().state_dict(), OUTPUT)


if __name__ == "__main__":
    main()