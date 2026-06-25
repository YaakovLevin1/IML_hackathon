import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    בלוק בסיסי של ResNet.
    מכיל שתי שכבות קונבולוציה וחיבור עוקף (Skip Connection).
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # שכבה ראשונה - יכולה להקטין את הממד המרחבי אם stride > 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # שכבה שנייה - שומרת על הממד
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # החיבור העוקף (Skip Connection)
        self.shortcut = nn.Sequential()
        # אם שינינו את גודל התמונה או את מספר הערוצים, אנחנו צריכים להתאים גם את החיבור העוקף
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # חיבור הקלט המקורי לפלט
        out = self.relu(out)
        return out


class ModelArchitecture(nn.Module):
    """
    Student model architecture (Custom Mini-ResNet built from scratch).
    """
    IMAGE_SIZE = 128  # גודל קטן יותר לאימון סופר-מהיר

    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.in_channels = 32

        # שכבת כניסה
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # בניית השכבות (כל אחת מקטינה את הרזולוציה פי 2 ומכפילה את מספר הערוצים)
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(256, stride=2)

        # שכבת פלט
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # מועך את הממדים המרחביים ל-1x1
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, stride):
        layer = ResidualBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args: x: batch of images
        Returns: logits for 20 classes
        """
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x