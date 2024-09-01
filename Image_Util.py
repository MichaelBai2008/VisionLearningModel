import matplotlib.pyplot as plt
import torch

from Config import Config
from pathlib import Path
from PIL import Image
from torchvision import transforms


def load_img_tensor(config: Config, img_file_path: Path) -> torch.tensor:
    # Load image from file
    img = Image.open(img_file_path)  # .convert("RGB")

    # Convert to tensor
    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(img)

    # resize base on config
    resize = transforms.Resize(size=(config.img_h_size, config.img_w_size))
    img_tensor = resize(img_tensor)

    # If original image is a GrayScale, simplly do R=G=B=GrayScale, otherewise, keep as is.
    img_tensor = img_tensor.expand(3, -1, -1)
    return img_tensor


def show_img_tensor_CHW(img_tensor: torch.tensor):
    plt.imshow(img_tensor.permute(1, 2, 0))  # C x H x W => H x W x C
