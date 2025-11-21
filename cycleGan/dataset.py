import os
import torch
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MonetPairedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_pairs = []

        for dataset_name in ['monet_style_dataset_A', 'monet_style_dataset_B']:
            dataset_path = os.path.join(root_dir, dataset_name)
            if not os.path.exists(dataset_path):
                continue

            original_dir = os.path.join(dataset_path, f'original_{dataset_name[-1]}')
            stylized_dir = os.path.join(dataset_path, f'stylized_{dataset_name[-1]}')

            if os.path.exists(original_dir) and os.path.exists(stylized_dir):
                original_images = sorted([f for f in os.listdir(original_dir)
                                        if f.endswith(('.jpg', '.png', '.jpeg'))])
                stylized_images = sorted([f for f in os.listdir(stylized_dir)
                                        if f.endswith(('.jpg', '.png', '.jpeg'))])

                for orig_img in original_images:
                    if orig_img in stylized_images:
                        self.image_pairs.append({
                            'original': os.path.join(original_dir, orig_img),
                            'stylized': os.path.join(stylized_dir, orig_img)
                        })
        print(f"Found {len(self.image_pairs)} paired images")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]

        original = Image.open(pair['original']).convert('RGB')
        stylized = Image.open(pair['stylized']).convert('RGB')

        if self.transform:
            original = self.transform(original)
            stylized = self.transform(stylized)

        return original, stylized


class MonetUnpairedDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='original'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.images = []

        for dataset_name in ['monet_style_dataset_A', 'monet_style_dataset_B']:
            dataset_path = os.path.join(root_dir, dataset_name)
            if not os.path.exists(dataset_path):
                continue

            if mode == 'original':
                img_dir = os.path.join(dataset_path, f'original_{dataset_name[-1]}')
            else:
                img_dir = os.path.join(dataset_path, f'stylized_{dataset_name[-1]}')

            if os.path.exists(img_dir):
                imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                       if f.endswith(('.jpg', '.png', '.jpeg'))]
                self.images.extend(imgs)
        print(f"Found {len(self.images)} {mode} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
if __name__ == '__main__':
    pass