import random

import lightning as L
import torch
import torchvision.transforms.v2 as transforms
from torch import optim
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision

from datasets import DetectionDataModule
from models import DetectionTLNetwork
from utils import show_images_with_boxes

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
])


class Detector(L.LightningModule):
    
    def __init__(self, lr, momentum):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = DetectionTLNetwork()
        
        self.val_IoU = IntersectionOverUnion()
        self.val_mAP = MeanAveragePrecision()
    
    def training_step(self, batch, batch_idx):
        images, labels, boxes = batch
        labels_and_boxes = [dict(labels=ls, boxes=bs) for ls, bs in zip(labels, boxes)]
        losses = self.model(images, labels_and_boxes)
        loss = sum(loss for loss in losses.values())
        self.log("train_loss", loss.item())
        return loss
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"])
    
    def validation_step(self, batch, batch_idx):
        images, true_labels, true_boxes = batch
        true_labels_and_boxes = [dict(labels=ls, boxes=bs) for ls, bs in zip(true_labels, true_boxes)]
        results = self.model(images)
        self.val_IoU.update(results, true_labels_and_boxes)
        self.val_mAP.update(results, true_labels_and_boxes)
    
    def on_validation_epoch_end(self):
        IoU = self.val_IoU.compute()["iou"]
        mAP = self.val_mAP.compute()["map"]
        self.log("val_iou", IoU)
        self.log("val_map", mAP)
        self.val_IoU.reset()
        self.val_mAP.reset()


detector = Detector(0.0001, 0.9)
trainer = L.Trainer(max_epochs=1, log_every_n_steps=1)
datamodule = DetectionDataModule(IMAGE_SIZE, BATCH_SIZE)
trainer.fit(detector, datamodule=datamodule)

# detector = Detector.load_from_checkpoint("./checkpoints/epoch=399-step=15200.ckpt").to("cpu")

detector.eval()
datamodule.setup("test")
test_dataset = datamodule.test_dataset

idxs = random.sample(range(len(test_dataset)), k=16)
samples = [test_dataset[i] for i in idxs]
images = torch.stack([sample["image"] for sample in samples], dim=0)
results = detector.model(images)

true_boxes = []
pred_boxes = []
scores = []
for sample, result in zip(samples, results):
    k = min(len(result["boxes"]), len(sample["boxes"]))
    idx = torch.topk(result["scores"], k=k).indices
    pred_boxes.append(result["boxes"][idx])
    scores.append(result["scores"][idx])
    true_boxes.append(sample["boxes"])

show_images_with_boxes(images, pred_boxes, true_boxes, scores)
