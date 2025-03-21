import matplotlib.pyplot as plt
from torchvision import utils


def show_image(image, boxes):
    image = utils.draw_bounding_boxes(image, boxes)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
