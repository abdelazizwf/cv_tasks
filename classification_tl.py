import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional.classification import accuracy

import lightning as L


ROOT_PATH = "./datasets/classification/"
TRAIN_PATH = ROOT_PATH + "train/"
TEST_PATH = ROOT_PATH + "test/"
BATCH_SIZE = 32

weights = models.ResNet18_Weights.DEFAULT
model_transforms = weights.transforms

to_tensor = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

train_transforms = transforms.Compose([
    to_tensor,
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
])

train_dataset = ImageFolder(TRAIN_PATH, transform=train_transforms)
# test_dataset = ImageFolder(TEST_PATH, transform=model_transforms)

train_dataset, val_dataset = random_split(train_dataset, [0.85, 0.15])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=11)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=11)

model_ft = models.resnet18(weights)
num_features = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_features, 5)


class CNN(L.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = model_ft
    
    def forward(self, X):
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss.item())
        return loss
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss.item())
        acc = accuracy(y_hat, y, task="multiclass", num_classes=5)
        self.log("val_accuracy", acc)


model = CNN()
trainer = L.Trainer(max_epochs=100)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
