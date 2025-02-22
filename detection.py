import torch
from torch import optim
from torch.utils.data import Dataset
import torchvision.io as io
from torchvision import tv_tensors, utils, models
import torchvision.transforms.v2 as transforms

import matplotlib.pyplot as plt
import tqdm

from pathlib import Path


DEVICE = "cuda"
TRAIN_PATH = "./datasets/detection/train/"
TEST_PATH = "./datasets/detection/val/"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def show_image(image, boxes):
    image = utils.draw_bounding_boxes(image, boxes)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


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
        image = tv_tensors.Image(io.decode_image(image_name)).to(DEVICE)
        
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
        boxes = tv_tensors.BoundingBoxes(fnums, format="XYXY", canvas_size=size).to(DEVICE)
        
        sample = {
            "image": image,
            "labels": torch.tensor(labels, dtype=torch.int64).to(DEVICE),
            "boxes": boxes,
        }
        
        if self.transform:
            sample["image"], sample["boxes"] = self.transform(sample["image"], sample["boxes"])
        
        return sample


to_tensor = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(
        {tv_tensors.Image: torch.float32, tv_tensors.BoundingBoxes: torch.float32},
        scale=True
    ),
]).to(DEVICE)

train_transforms = transforms.Compose([
    to_tensor,
    transforms.Resize(IMAGE_SIZE),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=15),
]).to(DEVICE)

test_transforms = transforms.Compose([
    to_tensor,
    transforms.Resize(IMAGE_SIZE),
]).to(DEVICE)

train_dataset = DetectionDataset(TRAIN_PATH, train_transforms)
test_dataset = DetectionDataset(TEST_PATH, test_transforms)

model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

model.train()
for epoch in tqdm.trange(10):
    for sample in tqdm.tqdm(train_dataset):
        image = sample.pop("image").to(DEVICE)
        losses = model([image], [sample])
        
        optimizer.zero_grad()
        unified_loss = torch.sum(torch.tensor(list(losses.values()), requires_grad=True)).to(DEVICE)
        unified_loss.backward()
        optimizer.step()
    tqdm.tqdm.write(f"Loss @ {epoch}: {unified_loss.item()}")

model.eval()
sample = test_dataset[130]
image = sample["image"]
result = model([image])[0]
idx = result["scores"].argmax(dim=0)
show_image(image.cpu(), result["boxes"][idx].unsqueeze(dim=0))
