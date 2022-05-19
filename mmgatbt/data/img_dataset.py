import torch
import os
import torchvision
from torchvision import transforms
from PIL import Image
from data.helpers import get_transforms

class ImageDataset(object):
    def __init__(self, args, nodes_data):
        self.args = args               
        self.nodes_data = nodes_data
        self.num_images = len(nodes_data)

        self.transform = get_transforms(args)

    def __getitem__(self, idx):
        im_path = self.nodes_data['image'][idx]
        im_name = im_path.split("/")[-1]
        
        img = Image.open(
            os.path.join(self.args.imdir_path, im_name)
        ).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return self.num_images
