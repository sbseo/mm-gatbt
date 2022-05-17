import torch
import os
import cv2
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
        im_path = self.nodes_data['image'][idx]
        im_name = im_path.split("/")[-1]
        
        img = cv2.imread(os.path.join(os.path.join(self.args.imdir_path, im_name)))
        # width = int(img.shape[1] * .25)
        # height = int(img.shape[0] * .25)
        # img = cv2.resize(img, (width, height))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img =  Image.fromarray(img)

        # img = Image.open(
        #     os.path.join(self.args.imdir_path, im_name)
        # ).convert("RGB")
        # width, height = img.size
        # width, height = int(width * .25), int(height * .25)
        # img = img.resize((width, height))
        return self.transform(img)

    def __len__(self):
        return self.num_images
