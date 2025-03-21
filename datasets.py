from functools import partial
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import io, tv_tensors
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms


class ClassificationDataModule(L.LightningDataModule):
    
    def __init__(self, image_size, batch_size, train_transform=None, test_transform=None):
        super().__init__()
        
        self.image_size = image_size
        self.batch_size = batch_size
        
        to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(image_size),
        ])
        
        self.train_transform = to_tensor
        self.test_transform = to_tensor
        
        if train_transform is not None:
            self.train_transform = transforms.Compose([
                to_tensor, train_transform
            ])
        
        if test_transform is not None:
            self.test_transform = transforms.Compose([
                to_tensor, test_transform
            ])
        
        self.get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=11)
    
    def setup(self, stage):
        root_path = "./datasets/classification/"
        
        if stage == "fit":
            train_path = root_path + "train/"
            dataset = ImageFolder(train_path, transform=self.train_transform)
            self.train_dataset, self.val_dataset = random_split(dataset, (0.85, 0.15))
        
        if stage == "test":
            test_path = root_path + "val/"
            self.test_dataset = ImageFolder(test_path, transform=self.test_transform)
    
    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset)
    
    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset)


class DetectionDataset(Dataset):
    
    def __init__(self, path, transform=None):
        super().__init__()
        self.transform = transform
        self.image_names = []
        self.labels = {}

        path = Path(path)
        for i, dir in enumerate(path.iterdir()):
            if not dir.is_dir():
                continue
            self.image_names.extend([x for x in dir.iterdir() if x.match("*.jpg")])
            self.labels[dir.name] = i
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        
        image_name = self.image_names[index]
        image = tv_tensors.Image(io.decode_image(image_name))
        
        file_name = image_name.with_suffix(".txt")
        fnums = []
        with open(file_name) as f:
            for line in f.readlines():
                line = [int(x) for x in line.split(',')]
                size = line[:2]
                x1, y1, x2, y2 = line[2:]
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    line[4] += 1
                    line[5] += 1
                fnums.append(line[2:])
        
        cls = file_name.parent.name
        labels = [self.labels[cls]] * len(fnums)
        boxes = tv_tensors.BoundingBoxes(fnums, format="XYXY", canvas_size=size)
        
        sample = {
            "image": image,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": boxes,
        }
        
        if self.transform:
            sample["image"], sample["boxes"] = self.transform(sample["image"], sample["boxes"])
        
        return sample
    
    @staticmethod
    def collate_fn(batch):
        images, labels, boxes = [], [], []
        for sample in batch:
            images.append(sample["image"])
            labels.append(sample["labels"])
            boxes.append(sample["boxes"])
        images = torch.stack(images, dim=0)
        return images, labels, boxes


class DetectionDataModule(L.LightningDataModule):
    
    def __init__(self, image_size, batch_size, train_transform=None, test_transform=None):
        super().__init__()
        
        self.image_size = image_size
        self.batch_size = batch_size
        
        to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(
                {tv_tensors.Image: torch.float32, tv_tensors.BoundingBoxes: torch.float32},
                scale=True
            ),
            transforms.Resize(image_size),
        ])
        
        self.train_transform = to_tensor
        self.test_transform = to_tensor
        
        if train_transform is not None:
            self.train_transform = transforms.Compose([
                to_tensor, train_transform
            ])
        
        if test_transform is not None:
            self.test_transform = transforms.Compose([
                to_tensor, test_transform
            ])
        
        self.get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=11, collate_fn=DetectionDataset.collate_fn)
    
    def setup(self, stage):
        root_path = "./datasets/detection/"
        
        if stage == "fit":
            train_path = root_path + "train/"
            dataset = DetectionDataset(train_path, self.train_transform)
            self.train_dataset, self.val_dataset = random_split(dataset, (0.85, 0.15))
        
        if stage == "test":
            test_path = root_path + "val/"
            self.test_dataset = DetectionDataset(test_path, self.test_transform)
    
    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=False)
    
    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset)
    
    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset)
