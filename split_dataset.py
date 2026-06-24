import os
import shutil
import torch
from torchvision import datasets
from torch.utils.data import random_split

# --- Configuration ---
SEED = 67
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
FINAL_TEST_RATIO = 0.15

# Define your base paths
base_dir = os.path.join('dataset', 'train_set')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

def main():
    # Set the seed for reproducibility
    torch.manual_seed(SEED)

    # 1. Load the dataset structure from the current train directory
    # This automatically finds all class folders and images
    print("Parsing dataset structure...")
    dataset = datasets.ImageFolder(train_dir)

    # 2. Calculate sizes based on the specified ratios
    total_len = len(dataset)
    train_size = int(TRAIN_RATIO * total_len)
    test_size = int(TEST_RATIO * total_len)
    final_test_size = total_len - train_size - test_size

    print(f"Total images: {total_len}")
    print(f"Target split -> Train: {train_size}, Val: {test_size}, Test: {final_test_size}")

    # 3. Generate the random split
    train_subset, val_subset, test_subset = random_split(
        dataset, 
        [train_size, test_size, final_test_size]
    )

    # 4. Helper function to move files for a specific subset
    def move_files(subset, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        
        # Iterate through the indices assigned to this subset
        for idx in subset.indices:
            # Get the original file path and class index from the dataset
            file_path, class_idx = dataset.samples[idx]
            class_name = dataset.classes[class_idx]
            
            # Create the corresponding class directory in the target folder
            target_class_dir = os.path.join(target_dir, class_name)
            os.makedirs(target_class_dir, exist_ok=True)
            
            # Move the file out of 'train/' and into the target directory
            filename = os.path.basename(file_path)
            new_path = os.path.join(target_class_dir, filename)
            
            shutil.move(file_path, new_path)

    # 5. Execute the move for Validation and Test sets
    print("Moving validation files...")
    move_files(val_subset, val_dir)
    
    print("Moving test files...")
    move_files(test_subset, test_dir)
    
    print("Dataset split complete. Train files remain in the original directory.")

if __name__ == "__main__":
    main()