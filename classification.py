from functools import partial

import lightning as L
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from torchmetrics.classification import Accuracy

from datasets import ClassificationDataModule
from models import ClassificationNetwork, ClassificationTLNetwork

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EXPERIMENT_NAME = "/cv-project-tl"
TRACKING_URI = "http://localhost:5000"


get_mlf_logger = partial(
    L.pytorch.loggers.MLFlowLogger,
    experiment_name=EXPERIMENT_NAME,
    tracking_uri=TRACKING_URI,
    log_model=False,
)

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
])


class Classifier(L.LightningModule):
    def __init__(self, lr, momentum):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.network = ClassificationTLNetwork()
        
        self.test_losses = []
        self.test_accuracy = Accuracy(task="multiclass", num_classes=5)
        
        self.val_losses = []
        self.val_accuracy = Accuracy(task="multiclass", num_classes=5)
        
        self.train_losses = []
        self.train_accuracy = Accuracy(task="multiclass", num_classes=5)
        
        self.final_loss = -1
    
    def forward(self, x):
        return self.network(x)
    
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
            {"val_loss": mean_loss, "val_accuracy": self.val_accuracy.compute(), "hp_metric": mean_loss}, step=self.current_epoch
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


def optimize_parameters():
    
    def objective(params):
        classifier = Classifier(**params)
        
        with mlflow.start_run(nested=True) as run:
            mlf_logger = get_mlf_logger(run_id=run.info.run_id)
            trainer = L.Trainer(max_epochs=30, logger=mlf_logger)
            
            logged_params = {
                "epochs": trainer.max_epochs,
                "lr": classifier.hparams.lr,
                "momentum": classifier.hparams.momentum,
                "criterion": "CrossEntropy",
                "optimizer": "SGD",
            }
            
            mlflow.log_params(logged_params)
            datamodule = ClassificationDataModule(IMAGE_SIZE, BATCH_SIZE, train_transforms)
            trainer.fit(model=classifier, datamodule=datamodule)
            
        return {"loss": classifier.final_loss, "status": STATUS_OK, "model": classifier}

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
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
        datamodule = ClassificationDataModule(IMAGE_SIZE, BATCH_SIZE, train_transforms)
        trainer.test(model=model, datamodule=datamodule)
        
        mlflow.pytorch.log_model(model, "best_model")


def run():
    classifier = Classifier(0.00136, 0.9287)
    trainer = L.Trainer(logger=True, max_epochs=30)
    datamodule = ClassificationDataModule(IMAGE_SIZE, BATCH_SIZE, train_transforms)
    trainer.fit(classifier, datamodule=datamodule)
    trainer.test(classifier, datamodule=datamodule)


if __name__ == "__main__":
    run()
