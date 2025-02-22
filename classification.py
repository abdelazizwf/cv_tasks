import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import Accuracy

import mlflow
import lightning as L
import numpy as np
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK

from functools import partial


ROOT_PATH = "./datasets/classification/"
TRAIN_PATH = ROOT_PATH + "train/"
TEST_PATH = ROOT_PATH + "test/"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EXPERIMENT_NAME = "/cv-project-5"
TRACKING_URI = "http://localhost:5000"


get_mlf_logger = partial(
    L.pytorch.loggers.MLFlowLogger,
    experiment_name=EXPERIMENT_NAME,
    tracking_uri=TRACKING_URI,
    log_model=False,
)


def get_dataloaders():
    to_tensor = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    train_transforms = transforms.Compose([
        to_tensor,
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
    ])

    test_transforms = transforms.Compose([
        to_tensor,
        transforms.Resize(IMAGE_SIZE),
    ])

    train_dataset = ImageFolder(TRAIN_PATH, transform=train_transforms)
    test_dataset = ImageFolder(TEST_PATH, transform=test_transforms)

    train_dataset, val_dataset = random_split(train_dataset, [0.85, 0.15])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=11)
    
    return train_dataloader, val_dataloader, test_dataloader


class ConvBlock(nn.Module):
    def __init__(self, inp, l1, l2):
        super().__init__()
        
        conv2d = partial(nn.Conv2d, kernel_size=3, stride=1, padding="same")
        
        self.layers = nn.Sequential(
            conv2d(inp, l1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(l1),
            conv2d(l1, l2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(l2),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        return self.layers(x)


class CNN(L.LightningModule):
    def __init__(self, lr, momentum):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.layers = nn.Sequential(
            ConvBlock(3, 8, 16),
            ConvBlock(16, 32, 64),
            ConvBlock(64, 128, 256),
            nn.AvgPool2d(16),
            nn.Flatten(),
            nn.Linear(256, 5)
        )
        
        self.test_losses = []
        self.test_accuracy = Accuracy(task="multiclass", num_classes=5)
        
        self.val_losses = []
        self.val_accuracy = Accuracy(task="multiclass", num_classes=5)
        
        self.train_losses = []
        self.train_accuracy = Accuracy(task="multiclass", num_classes=5)
        
        self.final_loss = -1
    
    def forward(self, x):
        return self.layers(x)
    
    def f_pass(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        return y, y_hat, loss
    
    def training_step(self, batch, batch_idx):
        y, y_hat, loss = self.f_pass(batch)
        self.train_losses.append(loss)
        self.train_accuracy.update(y_hat, y)
        return loss
    
    def on_train_epoch_end(self):
        mean_loss = torch.mean(torch.stack(self.train_losses)).item()
        self.logger.log_metrics(
            {"train_loss": mean_loss, "train_accuracy": self.train_accuracy.compute()}, step=self.current_epoch
        )
    
    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum
        )
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        y, y_hat, loss = self.f_pass(batch)
        self.val_losses.append(loss)
        self.val_accuracy.update(y_hat, y)
    
    def on_validation_epoch_end(self):
        mean_loss = torch.mean(torch.stack(self.val_losses)).item()
        self.logger.log_metrics(
            {"val_loss": mean_loss, "val_accuracy": self.val_accuracy.compute()}, step=self.current_epoch
        )
        self.final_loss = mean_loss
    
    def test_step(self, batch, batch_idx):
        y, y_hat, loss = self.f_pass(batch)
        self.test_losses.append(loss)
        self.test_accuracy.update(y_hat, y)
    
    def on_test_end(self):
        mean_loss = torch.mean(torch.stack(self.test_losses)).item()
        self.logger.log_metrics(
            {"test_loss": mean_loss, "test_accuracy": self.test_accuracy.compute()}
        )


def objective(params):
    train_dataloader, val_dataloader, _ = get_dataloaders()
    model = CNN(**params)
    
    with mlflow.start_run(nested=True) as run:
        mlf_logger = get_mlf_logger(run_id=run.info.run_id)
        trainer = L.Trainer(max_epochs=30, logger=mlf_logger)
        
        logged_params = {
            "epochs": trainer.max_epochs,
            "lr": model.hparams.lr,
            "momentum": model.hparams.momentum,
            "criterion": "CrossEntropy",
            "optimizer": "SGD",
        }
        
        mlflow.log_params(logged_params)
        
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
    return {"loss": model.final_loss, "status": STATUS_OK, "model": model}


if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    _, _, test_dataloader = get_dataloaders()
    
    space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-2)),
        "momentum": hp.uniform("momentum", 0.9, 1.0),
    }
    
    with mlflow.start_run(run_name="finer-search-1") as run:
        trials = Trials()
        best_params = fmin(
            fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials
        )
        
        best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
        
        mlflow.log_params(best_params)
        mlflow.log_metric("val_loss", best_run["loss"])
    
        model = best_run["model"]
        mlf_logger = get_mlf_logger(run_id=run.info.run_id)
        trainer = L.Trainer(logger=mlf_logger)
        trainer.test(model=model, dataloaders=test_dataloader)
        
        mlflow.pytorch.log_model(model, "best_model")
