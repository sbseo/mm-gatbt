import torch
import os
import torchvision
from torchvision import transforms
from PIL import Image


class ImageDataset(object):
    def __init__(self, args, nodes_data):

        self.args = args               
        self.nodes_data = nodes_data
        self.num_images = len(nodes_data)

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.46777044, 0.44531429, 0.40661017],
                    std=[0.12221994, 0.12145835, 0.14380469],
                ),
            ]
        )

    def __getitem__(self, idx):

        img_path = self.nodes_data['image'][idx]
        img = Image.open(
            os.path.join(self.args.data_path, img_path)
        ).convert("RGB")
        img = self.transform(img)
        # img = torchvision.transforms.ToTensor()(img)
        # return torch.ones(3,7,7)
        return img

    def __len__(self):
        return self.num_images