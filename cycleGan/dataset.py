import os
import torch
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class Monet(Dataset):
    def __init__(self, root, is_train=True, transform=None):
        if is_train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        photo_path = os.path.join(root, "origin")
        style_path = os.path.join(root, "style")
        self.transform = transform
        self.length_dataset = len(os.listdir(photo_path))
        self.list_photo_path = [os.path.join(photo_path, x) for x in os.listdir(photo_path)]
        self.list_style_path = [os.path.join(style_path, x) for x in os.listdir(style_path)]


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        try:
            origin_img_path = self.list_photo_path[index % self.length_dataset]
            style_img_path = self.list_style_path[index % self.length_dataset]

            origin_img = Image.open(origin_img_path).convert("RGB")
            style_img = Image.open(style_img_path).convert("RGB")

            if self.transform:
                origin_img = self.transform(origin_img)
                style_img = self.transform(style_img)

            return origin_img, style_img

        except (FileNotFoundError, OSError) as e:
            print(f"Error loading images: {e}")
            next_index = (index + 1) % self.length_dataset
            return self.__getitem__(next_index)

    def cout(self, a):
        print(a)
        print(type(a))
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    train_dataset = Monet(
        root="C:/Users/tam/Documents/data/monet_style_dataset",
        is_train=True,
        transform=transform
    )
    print("Số lượng mẫu:", len(train_dataset))

    origin_img, style_img = train_dataset[10]

    # Chuyển tensor về numpy để hiển thị
    def show_tensor_img(tensor_img, title=""):
        img = tensor_img.permute(1, 2, 0).cpu().numpy()  # (C,H,W) -> (H,W,C)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    show_tensor_img(origin_img, "Origin")

    plt.subplot(1, 2, 2)
    show_tensor_img(style_img, "Style")

    plt.show()