import random

import lightning as L
import torch
from torch import nn, optim
from torchvision.transforms import v2 as T

from datasets import FewShotLearningDataModule
from models import SiameseNetwork
from utils import show_image_pair


def contrastive_loss(X1, X2, label, margin=2.0):
    distance = torch.nn.functional.pairwise_distance(X1, X2, keepdim=True)
    loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                      (label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    return loss


class FewShotLearner(L.LightningModule):
    
    def __init__(self, lr, momentum):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = SiameseNetwork()
        
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, X1, X2):
        return self.model(X1, X2)
    
    def f_pass(self, batch):
        X1, X2, label, *_ = batch
        out1, out2 = self(X1, X2)
        loss = contrastive_loss(out1, out2, label)
        return loss
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"])
    
    def training_step(self, batch, batch_idx):
        loss = self.f_pass(batch)
        self.train_losses.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        loss = torch.mean(torch.stack(self.train_losses)).item()
        self.log("train_loss", loss)
        self.train_losses.clear()
    
    def validation_step(self, batch, batch_idx):
        loss = self.f_pass(batch)
        self.val_losses.append(loss)
    
    def on_validation_epoch_end(self):
        loss = torch.mean(torch.stack(self.val_losses)).item()
        self.log("val_loss", loss)
        self.val_losses.clear()


learner = FewShotLearner(1e-3, 0.9)
datamodule = FewShotLearningDataModule(
    (256, 256), 16, train_transform=T.Grayscale(), test_transform=T.Grayscale()
)
trainer = L.Trainer(max_epochs=10)
trainer.fit(learner, datamodule=datamodule)

learner.eval()
datamodule.setup("test")
test_dataset = datamodule.test_dataset

for _ in range(10):
    image1, image2, label, class1, class2 = random.choice(test_dataset)
    y1, y2 = learner.model(image1.unsqueeze(dim=0), image2.unsqueeze(dim=0))
    distance = nn.functional.pairwise_distance(y1, y2)
    s = f"image1={class1}  image2={class2}  distance={distance.item()}"
    show_image_pair(image1, image2, s)
