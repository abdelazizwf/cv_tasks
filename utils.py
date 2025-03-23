from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
from torchvision import utils


def show_images_with_boxes(
    images: Iterable[torch.Tensor],
    pred_boxes: Iterable[torch.Tensor],
    true_boxes: Iterable[torch.Tensor],
    scores: Iterable[torch.Tensor],
    figsize: tuple[int, int] = (20, 20)
):
    n_images = len(images)
    ncols = 4
    nrows = n_images // ncols
    nrows = nrows + 1 if n_images % ncols != 0 else nrows
    
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    
    for ax, image, pred_box, true_box, score in zip(axes.flat, images, pred_boxes, true_boxes, scores):
        labels = [""] * len(true_box)
        colors = ["green"] * len(true_box)
        for s in score:
            labels.append(f"{s:.2f}")
            colors.append("yellow")
        image = utils.draw_bounding_boxes(
            image=image,
            boxes=torch.stack([*true_box, *pred_box]),
            labels=labels,
            colors=colors,
        )
        ax.imshow(image.permute(1, 2, 0))
    
    plt.tight_layout()
    plt.show()
