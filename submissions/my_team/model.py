import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    """
    Student model architecture.

    Students should define their model here.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """
    IMAGE_SIZE = 243

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 7, 2, 3) # 243 -> 122
        self.pool1 = nn.MaxPool2d(2, 2) # 122 -> 61
        self.conv2 = nn.Conv2d(32, 64, 5, 2, 2) # 61 -> 31
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1) # 31 -> 16
        self.pool2 = nn.MaxPool2d(2, 2) # 16 -> 8
        self.fc1 = nn.Linear(8*8*128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 20)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: batch of images

        Returns:
            logits for 20 classes
        """

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x