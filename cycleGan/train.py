import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset import MonetUnpairedDataset, MonetPairedDataset
from model import Generator, Discriminator
from PIL import Image
import torchvision.transforms as transforms

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CycleGANLoss:
    """Loss functions for CycleGAN"""
    def __init__(self, device, lambda_cycle=10):
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_cycle = nn.L1Loss()

    def discriminator_loss(self, real_output, fake_output):
        """Discriminator loss"""
        real_loss = self.criterion_GAN(real_output, torch.ones_like(real_output))
        fake_loss = self.criterion_GAN(fake_output, torch.zeros_like(fake_output))
        total_loss = (real_loss + fake_loss) * 0.5
        return total_loss

    def generator_loss(self, fake_output):
        """Generator adversarial loss"""
        return self.criterion_GAN(fake_output, torch.ones_like(fake_output))

    def cycle_consistency_loss(self, real_image, cycled_image):
        """Cycle consistency loss"""
        return self.criterion_cycle(cycled_image, real_image) * self.lambda_cycle

class CycleGAN:
    def __init__(self, device):
        self.device = device
        
        # Initialize models
        self.G_monet = Generator().to(device)  # Photo -> Monet
        self.G_photo = Generator().to(device)  # Monet -> Photo
        self.D_monet = Discriminator().to(device)
        self.D_photo = Discriminator().to(device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            list(self.G_monet.parameters()) + list(self.G_photo.parameters()),
            lr=2e-4, betas=(0.5, 0.999)
        )
        self.optimizer_D_monet = optim.Adam(self.D_monet.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D_photo = optim.Adam(self.D_photo.parameters(), lr=2e-4, betas=(0.5, 0.999))
        
        # Loss function
        self.loss_fn = CycleGANLoss(device, LAMBDA_CYCLE)
        
        # History
        self.history = {
            'g_monet_loss': [], 'g_photo_loss': [],
            'd_monet_loss': [], 'd_photo_loss': []
        }
    
    def train_step(self, real_photo, real_monet):
        
        # ==================== Train Generators ====================
        self.optimizer_G.zero_grad()
        
        # GAN loss
        fake_monet = self.G_monet(real_photo)
        pred_fake_monet = self.D_monet(fake_monet)
        loss_GAN_monet = self.loss_fn.generator_loss(pred_fake_monet)
        
        fake_photo = self.G_photo(real_monet)
        pred_fake_photo = self.D_photo(fake_photo)
        loss_GAN_photo = self.loss_fn.generator_loss(pred_fake_photo)
        
        # Cycle consistency loss
        cycled_monet = self.G_monet(fake_photo)
        loss_cycle_monet = self.loss_fn.cycle_consistency_loss(real_monet, cycled_monet)
        
        cycled_photo = self.G_photo(fake_monet)
        loss_cycle_photo = self.loss_fn.cycle_consistency_loss(real_photo, cycled_photo)
        
        # Total generator loss
        loss_G = (loss_GAN_monet + loss_GAN_photo + 
                  loss_cycle_monet + loss_cycle_photo)
        
        loss_G.backward()
        self.optimizer_G.step()
        
        # ==================== Train Discriminator Monet ====================
        self.optimizer_D_monet.zero_grad()
        
        pred_real_monet = self.D_monet(real_monet)
        pred_fake_monet = self.D_monet(fake_monet.detach())
        loss_D_monet = self.loss_fn.discriminator_loss(pred_real_monet, pred_fake_monet)
        
        loss_D_monet.backward()
        self.optimizer_D_monet.step()
        
        # ==================== Train Discriminator Photo ====================
        self.optimizer_D_photo.zero_grad()
        
        pred_real_photo = self.D_photo(real_photo)
        pred_fake_photo = self.D_photo(fake_photo.detach())
        loss_D_photo = self.loss_fn.discriminator_loss(pred_real_photo, pred_fake_photo)
        
        loss_D_photo.backward()
        self.optimizer_D_photo.step()
        
        return {
            'g_monet_loss': loss_GAN_monet.item() + loss_cycle_monet.item(),
            'g_photo_loss': loss_GAN_photo.item() + loss_cycle_photo.item(),
            'd_monet_loss': loss_D_monet.item(),
            'd_photo_loss': loss_D_photo.item()
        }
    
    def train(self, photo_loader, monet_loader, epochs):
        """Training loop"""
        for epoch in range(epochs):
            epoch_losses = {'g_monet_loss': [], 'g_photo_loss': [], 
                           'd_monet_loss': [], 'd_photo_loss': []}
            
            # Training
            self.G_monet.train()
            self.G_photo.train()
            self.D_monet.train()
            self.D_photo.train()
            
            for i, (real_photo, real_monet) in enumerate(zip(photo_loader, monet_loader)):
                real_photo = real_photo.to(self.device)
                real_monet = real_monet.to(self.device)
                
                losses = self.train_step(real_photo, real_monet)
                
                for key in epoch_losses:
                    epoch_losses[key].append(losses[key])
                
                if (i + 1) % 50 == 0:
                    print(f"  Batch [{i+1}/{len(photo_loader)}] "
                          f"G_monet: {losses['g_monet_loss']:.4f} "
                          f"D_monet: {losses['d_monet_loss']:.4f}")
            
            # Average losses for epoch
            for key in epoch_losses:
                avg_loss = np.mean(epoch_losses[key])
                self.history[key].append(avg_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"G_monet: {self.history['g_monet_loss'][-1]:.4f} "
                  f"G_photo: {self.history['g_photo_loss'][-1]:.4f} "
                  f"D_monet: {self.history['d_monet_loss'][-1]:.4f} "
                  f"D_photo: {self.history['d_photo_loss'][-1]:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_models(f'cyclegan_epoch_{epoch+1}.pth')
    
    def save_models(self, path='cyclegan_models.pth'):
        """Save model weights"""
        torch.save({
            'G_monet': self.G_monet.state_dict(),
            'G_photo': self.G_photo.state_dict(),
            'D_monet': self.D_monet.state_dict(),
            'D_photo': self.D_photo.state_dict(),
        }, path)
        print(f"Models saved to {path}")
    
    def load_models(self, path='cyclegan_models.pth'):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.G_monet.load_state_dict(checkpoint['G_monet'])
        self.G_photo.load_state_dict(checkpoint['G_photo'])
        self.D_monet.load_state_dict(checkpoint['D_monet'])
        self.D_photo.load_state_dict(checkpoint['D_photo'])
        print(f"Models loaded from {path}")


def visualize_dataset_samples(dataset, samples=8):
    """Visualize paired samples from dataset"""
    fig, axes = plt.subplots(2, samples, figsize=(3*samples, 6))
    
    for i in range(samples):
        if i >= len(dataset):
            break
        original, stylized = dataset[i]
        
        # Denormalize
        orig = original.permute(1, 2, 0).numpy()
        orig = (orig * 0.5 + 0.5).clip(0, 1)
        
        style = stylized.permute(1, 2, 0).numpy()
        style = (style * 0.5 + 0.5).clip(0, 1)
        
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(style)
        axes[1, i].set_title(f'Monet Style {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_results(model, photo_loader, samples=5):
    """Visualize photo-to-Monet transformation"""
    model.G_monet.eval()
    
    fig, axes = plt.subplots(samples, 2, figsize=(8, 4*samples))
    if samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, img in enumerate(photo_loader):
            if i >= samples:
                break
            
            img = img.to(model.device)
            fake_monet = model.G_monet(img)
            
            # Original photo
            orig = img[0].permute(1, 2, 0).cpu().numpy()
            orig = (orig * 0.5 + 0.5).clip(0, 1)
            
            # Generated Monet
            gen = fake_monet[0].permute(1, 2, 0).cpu().numpy()
            gen = (gen * 0.5 + 0.5).clip(0, 1)
            
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title('Input Photo')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gen)
            axes[i, 1].set_title('Monet-esque')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_losses(history):
    """Plot training losses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Generator losses
    ax1.plot(history['g_monet_loss'], label='Monet Generator', linewidth=2)
    ax1.plot(history['g_photo_loss'], label='Photo Generator', linewidth=2)
    ax1.set_title('Generator Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Discriminator losses
    ax2.plot(history['d_monet_loss'], label='Monet Discriminator', linewidth=2)
    ax2.plot(history['d_photo_loss'], label='Photo Discriminator', linewidth=2)
    ax2.set_title('Discriminator Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set your dataset path
    DATASET_ROOT = '/kaggle/input/paired-landscape-and-monetstylised-image/monet_style_dataset'
    set_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    IMAGE_SIZE = 256
    CHANNELS = 3
    EPOCHS = 30
    BATCH_SIZE = 4  # Increased from 1 for better training
    LAMBDA_CYCLE = 10
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    print("="*60)
    print("Loading Datasets...")
    print("="*60)
    # Option 1: Use unpaired datasets (original CycleGAN approach)
    photo_dataset = MonetUnpairedDataset(DATASET_ROOT, transform=transform, mode='original')
    monet_dataset = MonetUnpairedDataset(DATASET_ROOT, transform=transform, mode='stylized')
    
    photo_loader = DataLoader(photo_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    monet_loader = DataLoader(monet_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Option 2: Use paired dataset (for visualization/evaluation)
    paired_dataset = MonetPairedDataset(DATASET_ROOT, transform=transform)
    paired_loader = DataLoader(paired_dataset, batch_size=1, shuffle=False)
    
    print("\n" + "="*60)
    print("Visualizing Sample Data...")
    print("="*60)
    visualize_dataset_samples(paired_dataset, samples=6)
    
    print("\n" + "="*60)
    print("Initializing CycleGAN Model...")
    print("="*60)
    cyclegan = CycleGAN(device)
    
    print(f"\nGenerator parameters: {sum(p.numel() for p in cyclegan.G_monet.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in cyclegan.D_monet.parameters()):,}")
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    cyclegan.train(photo_loader, monet_loader, epochs=EPOCHS)
    
    # Plot training history
    plot_losses(cyclegan.history)
    
    # Visualize results
    visualize_results(cyclegan, photo_loader, samples=5)
    
    # Save final model
    cyclegan.save_models('cyclegan_monet_final.pth')
    
    print("\n" + "="*60)
    print("Setup Complete! Ready to train.")
    print("Uncomment the training lines above to start.")
    print("="*60)