from datetime import datetime
from pathlib import Path
import json
import re
import random
import argparse
import os

import joblib
from nbformat import write
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

from base_model import ImageNetSubset
from submissions.my_team.model import ModelArchitecture

PRINT_STATS = True
try:
    import psutil
    import GPUtil
except:
    PRINT_STATS = False

# Define device globally or in main() - standard practice is to resolve it dynamically
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

DATA_ROOT = Path("dataset")
LABELS_LIST = Path("dataset/labels.json")
OUTPUT = Path("weights.joblib")
OUTPUT_LOG = "logs/training" #_{}.log"
CHECKPOINT_DIR = Path("checkpoints")

SEED = 67
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
FINAL_TEST_RATIO = 0.15

BATCH_SIZE = 32


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def save_preview_images(data_loader, num_images=10, output_path="preview_augmentations.png"):
    """
    Grabs a batch of images from the dataloader and saves them to a file 
    so you can visually inspect the augmentations.
    """
    # Grab the first batch of images and labels
    images, labels = next(iter(data_loader))
    
    # Ensure we don't try to grab more images than exist in a single batch
    num_images = min(num_images, images.size(0))
    images_to_save = images[:num_images]
    
    # Save the images as a grid (nrow=5 means 5 images per row)
    save_image(images_to_save, output_path, nrow=5)
    print(f"Saved a preview of {num_images} augmented images to {output_path}")

class AddRectangles:
    def __init__(self, num_rectangles=3, max_size=50, color=(0,0,0)):
        self.num_rectangles = num_rectangles
        self.max_size = max_size
        self.color = color

    def __call__(self, img):
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(self.num_rectangles):
            rw = random.randint(10, self.max_size)
            rh = random.randint(10, self.max_size)
            x = random.randint(0, w - rw)
            y = random.randint(0, h - rh)
            draw.rectangle([x, y, x + rw, y + rh], fill=self.color)
        return img

class AddBalls:
    def __init__(self, num_balls=3, max_radius=20, color=(0,0,0)):
        self.num_balls = num_balls
        self.max_radius = max_radius
        self.color = color

    def __call__(self, img):
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(self.num_balls):
            r = random.randint(5, self.max_radius)
            x = random.randint(r, w - r)
            y = random.randint(r, h - r)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=self.color)
        return img

class WeightedRandomAugmentations:
    def __init__(self, transforms, count_weights):
        """
        count_weights: list where index 0 = P(apply 0), index 1 = P(apply 1), index 2 = P(apply 2), etc.
        e.g. [0.2, 0.55, 0.15, 0.1] → 20% chance of no augmentation, 55% chance of 1, 15% of 2, 10% of 3
        """
        self.transforms = transforms
        self.counts = range(1, len(count_weights) + 1)
        self.weights = count_weights

    def __call__(self, img):
        n = random.choices(self.counts, weights=self.weights, k=1)[0]
        chosen = random.sample(self.transforms, k=min(n, len(self.transforms)))
        for t in chosen:
            img = t(img)
        return img

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(ModelArchitecture.IMAGE_SIZE),
    transforms.CenterCrop(ModelArchitecture.IMAGE_SIZE),
    transforms.ToTensor(),
])

IMAGE_TRANSFORMS_ALL_AUGMENTATIONS = transforms.Compose([
    transforms.Resize(ModelArchitecture.IMAGE_SIZE),
    transforms.CenterCrop(ModelArchitecture.IMAGE_SIZE),
    AddRectangles(),
    AddBalls(),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor(),
])

IMAGE_TRANSFORMS_RANDOM_AUGMENTATIONS = transforms.Compose([
    transforms.Resize(ModelArchitecture.IMAGE_SIZE),
    transforms.CenterCrop(ModelArchitecture.IMAGE_SIZE),

    WeightedRandomAugmentations(
        transforms=[
            AddRectangles(),
            AddBalls(),
            transforms.RandomRotation(degrees=180),
        ],
        count_weights=[0.2, 0.55, 0.15, 0.1],  # 20% → 0, 55% → 1, 15% → 2, 10% → 3
    ),

    transforms.ToTensor(),
])

IMAGE_TRANSFORMS_RANDOM_AUGMENTATIONS_FASTER_BUT_WORSE = transforms.Compose([
    transforms.Resize(ModelArchitecture.IMAGE_SIZE),
    transforms.CenterCrop(ModelArchitecture.IMAGE_SIZE),
    transforms.RandomRotation(degrees=180),
    
    # CRITICAL: Convert to Tensor FIRST before applying math-based augmentations
    transforms.ToTensor(), 
    
    # RandomErasing replaces "AddRectangles" and "AddBalls". 
    # It randomly drops black (or colored) rectangles onto the tensor.
    # We apply it multiple times sequentially to simulate the "num_rectangles=3" effect.
    transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), value=0)], p=0.5),
    transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), value=0)], p=0.5),
    transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), value=0)], p=0.3),
])

def calculate_accuracy(model, data_loader, device):
    """
    Calculate the accuracy of the model on the provided data loader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            # MOVED TO DEVICE
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def train_one_epoch(epoch_index, tb_writer, model, optimizer, scheduler, train_loader, device, report_interval=5):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    last_loss = 0.0
    for batch_index, (images, labels) in enumerate(train_loader):
        # MOVED TO DEVICE
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        scheduler.step()

        running_loss += loss.item()
        if batch_index % report_interval == report_interval - 1:  # Log every `report_interval` batches
            last_loss = running_loss / report_interval
            timestamp = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
            print(f"{timestamp}: Epoch [{epoch_index + 1}], Batch [{batch_index + 1}], Loss: {last_loss:.4f}")
            tb_writer.add_scalar('training loss', last_loss, epoch_index * len(train_loader) + batch_index)
            running_loss = 0.0

            if PRINT_STATS:
                cpu_util = psutil.cpu_percent()
                gpus = GPUtil.getGPUs()
                gpu_util = gpus[0].load * 100 if gpus else 0
                
                # --- ADD THIS: Log to TensorBoard ---
                tb_writer.add_scalar('Hardware/CPU_Utilization', cpu_util, epoch_index * len(train_loader) + batch_index)
                tb_writer.add_scalar('Hardware/GPU_Utilization', gpu_util, epoch_index * len(train_loader) + batch_index)


    return last_loss

def main(args):
    """
    Full training pipeline.
    This script must create weights.joblib.
    """
    global BATCH_SIZE

    BATCH_SIZE = args.batch_size
    print(f"Batch size is {BATCH_SIZE}")

    print(f"Using device: {device}")
    model = ModelArchitecture().to(device)

    # --- RESUME FROM CHECKPOINT LOGIC ---
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            if args.resume.endswith('.joblib'):
                state_dict = joblib.load(args.resume)
            else:
                state_dict = torch.load(args.resume, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint path '{args.resume}' does not exist. Starting from scratch.")

    labels_list = json.load(open(LABELS_LIST)) # str -> str
    labels_list = {int(k): v for k, v in labels_list.items()} # int -> str

    # initialize seed
    set_seed(SEED)

    train_dataset = ImageNetSubset(DATA_ROOT,
                                   split="train_set/train",
                                   transform=IMAGE_TRANSFORMS_RANDOM_AUGMENTATIONS_FASTER_BUT_WORSE)
    validation_dataset = ImageNetSubset(DATA_ROOT,
                                        split="train_set/validation",
                                        transform=IMAGE_TRANSFORMS_RANDOM_AUGMENTATIONS_FASTER_BUT_WORSE)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Ensure logs and checkpoint directories exist
    os.makedirs("logs", exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    writer = SummaryWriter(OUTPUT_LOG) #.format(timestamp))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # weight_decay is a type of regularization
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # FIXED: Generator device is now dynamically assigned based on the active hardware
    gen = torch.Generator(device='cpu')
    
    # load the data
    # Dynamically calculate workers based on available CPU cores.
    # We cap at 16 because anything higher on a single GPU usually causes overhead bottlenecks.
    cpu_cores = os.cpu_count()
    NUM_WORKERS = min(16, cpu_cores) if cpu_cores is not None else 4
    print(f"System has {cpu_cores} CPU cores. Using {NUM_WORKERS} workers for DataLoaders.")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        generator=gen, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        validation_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        generator=gen, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3, # The peak learning rate
        steps_per_epoch=len(train_loader),
        epochs=args.epochs, # Use your command line argument here!
        pct_start=0.1
    )

    # previw a few images
    save_preview_images(train_loader, num_images=10, output_path="preview_augmentations.png")

    # --- TRAINING LOOP WITH CHECKPOINTING ---
    for epoch in range(args.epochs):
        train_one_epoch(epoch, writer, model, optimizer, scheduler, train_loader, device)
        
        accuracy = calculate_accuracy(model, val_loader, device)
        print(f"Epoch [{epoch + 1}] completed. Validation Accuracy: {accuracy:.2f}%")
        writer.add_scalar('validation accuracy', accuracy, epoch)
        
        # --- Check LR and Step Scheduler ---
        # current_lr = optimizer.param_groups[0]['lr']
        # scheduler.step(accuracy) # This will drop the LR if accuracy stops improving
        # new_lr = optimizer.param_groups[0]['lr']
        
        # if current_lr != new_lr:
        #     print(f"Learning rate reduced from {current_lr} to {new_lr}")

        # Save intermediate checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved epoch {epoch + 1} checkpoint to {checkpoint_path}")

    # evaluate the model on the test set
    accuracy = calculate_accuracy(model, val_loader, device)
    print(f"Final Validation Accuracy: {accuracy:.2f}%")

    # write to file
    writer.flush()
    writer.close()
    
    # Save final joblib payload
    joblib.dump(model.state_dict(), OUTPUT)
    print(f"Saved trained {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on ImageNet subset")
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train the model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint file to resume training from')
    
    args = parser.parse_args()
    main(args)