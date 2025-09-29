import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset import Monet
from model import Generator, Discriminator
from PIL import Image
from torch.utils.data import Dataset

def save_checkpoint(filepath, epoch, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, best_score=None):
    # Handle DataParallel wrapper
    def get_state_dict(model):
        if hasattr(model, 'module'):
            return model.module.state_dict()
        return model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'G_XtoY_state_dict': get_state_dict(G_XtoY),
        'G_YtoX_state_dict': get_state_dict(G_YtoX),
        'D_X_state_dict': get_state_dict(D_X),
        'D_Y_state_dict': get_state_dict(D_Y),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'best_score': best_score
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, device):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)

        # Handle DataParallel wrapper
        def load_state_dict(model, state_dict):
            if hasattr(model, 'module'):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

        load_state_dict(G_XtoY, checkpoint['G_XtoY_state_dict'])
        load_state_dict(G_YtoX, checkpoint['G_YtoX_state_dict'])
        load_state_dict(D_X, checkpoint['D_X_state_dict'])
        load_state_dict(D_Y, checkpoint['D_Y_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        best_score = checkpoint.get('best_score', float('inf'))
        return checkpoint['epoch'], best_score
    return 0, float('inf')

def train():
    batch_size = 4
    lr = 1e-4
    num_epochs = 100
    image_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multi-GPU setup
    num_gpus = torch.cuda.device_count()

    out_path = "out1"
    os.makedirs(out_path, exist_ok=True)

    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = Monet(root="C:/Users/tam/Documents/data/monet_style_dataset", is_train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    val_dataset = Monet(root="C:/Users/tam/Documents/data/monet_style_dataset", is_train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )

    G_XtoY = Generator().to(device)
    G_YtoX = Generator().to(device)
    D_X = Discriminator().to(device)
    D_Y = Discriminator().to(device)

    # Wrap models with DataParallel for multi-GPU training
    if num_gpus > 1:
        G_XtoY = nn.DataParallel(G_XtoY)
        G_YtoX = nn.DataParallel(G_YtoX)
        D_X = nn.DataParallel(D_X)
        D_Y = nn.DataParallel(D_Y)
        print("Models wrapped with DataParallel")

    g_optimizer = torch.optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=lr, betas=(0.5, 0.99))
    d_optimizer = torch.optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()), lr=lr, betas=(0.5, 0.99))

    adversarial_loss = nn.BCEWithLogitsLoss()
    cycle_loss = nn.L1Loss()

    start_epoch, best_score = load_checkpoint(os.path.join(out_path, 'cyclegan_checkpoint.pt'), G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, device)

    writer = SummaryWriter(os.path.join(out_path, 'tensorboard_logs'))

    neutral_img, happy_img = next(iter(val_dataloader))
    neutral_img = neutral_img.to(device)
    happy_img = happy_img.to(device)

    def calculate_image_similarity(img1, img2):
        # Simple MSE-based similarity (lower is better, so we return 1-MSE for higher=better)
        mse = F.mse_loss(img1, img2).item()
        return 1.0 / (1.0 + mse)  # Convert to similarity score (0-1, higher is better)

    for epoch in range(start_epoch, num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (x_real, y_real) in enumerate(progress_bar):
            x_real, y_real = x_real.to(device), y_real.to(device)

            # Train D
            y_fake = G_XtoY(x_real)
            x_fake = G_YtoX(y_real)

            d_loss_x = adversarial_loss(D_X(x_real), torch.ones_like(D_X(x_real))) + \
                       adversarial_loss(D_X(x_fake.detach()), torch.zeros_like(D_X(x_fake)))
            d_loss_y = adversarial_loss(D_Y(y_real), torch.ones_like(D_Y(y_real))) + \
                       adversarial_loss(D_Y(y_fake.detach()), torch.zeros_like(D_Y(y_fake)))

            d_loss = (d_loss_x + d_loss_y) / 2
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train G
            y_fake = G_XtoY(x_real)
            x_fake = G_YtoX(y_real)

            x_recon = G_YtoX(y_fake)
            y_recon = G_XtoY(x_fake)
            g_loss_x = adversarial_loss(D_X(x_fake), torch.ones_like(D_X(x_fake)))
            g_loss_y = adversarial_loss(D_Y(y_fake), torch.ones_like(D_Y(y_fake)))

            cycle_loss_x = cycle_loss(x_recon, x_real)
            cycle_loss_y = cycle_loss(y_recon, y_real)

            # Identity loss: G_XtoY(y) should be close to y, G_YtoX(x) should be close to x
            identity_loss_x = cycle_loss(G_YtoX(x_real), x_real)  # G_YtoX(x) vs x
            identity_loss_y = cycle_loss(G_XtoY(y_real), y_real)  # G_XtoY(y) vs y

            g_loss = g_loss_x + g_loss_y + 10 * (cycle_loss_x + cycle_loss_y) + 5 * (identity_loss_x + identity_loss_y)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                'D_Loss': f'{d_loss.item():.4f}',
                'G_Loss': f'{g_loss.item():.4f}'
            })

            global_step = epoch * len(train_dataloader) + i
            if i % 50 == 0:
                writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
                writer.add_scalar('Loss/Cycle_X', cycle_loss_x.item(), global_step)
                writer.add_scalar('Loss/Cycle_Y', cycle_loss_y.item(), global_step)
                writer.add_scalar('Loss/Identity_X', identity_loss_x.item(), global_step)
                writer.add_scalar('Loss/Identity_Y', identity_loss_y.item(), global_step)

            # Evaluation step
            if (i % 100 == 0) and (i != 0):
                with torch.no_grad():
                    paired_images = []

                    for j in range(neutral_img.shape[0]):
                        x_real_img = neutral_img[j].unsqueeze(0)
                        x_fake_img = G_XtoY(x_real_img)
                        paired_images.append(torch.cat([x_real_img, x_fake_img], dim=3))
                    for j in range(happy_img.shape[0]):
                        y_real_img = happy_img[j].unsqueeze(0)
                        y_fake_img = G_YtoX(y_real_img)
                        paired_images.append(torch.cat([y_real_img, y_fake_img], dim=3))

                    x_concat = torch.cat(paired_images, dim=2)

                    x_concat = ((x_concat + 1) / 2).clamp_(0, 1)

                    sample_path = os.path.join(out_path, f'epoch{epoch}_step{i}-images.jpg')
                    save_image(x_concat, sample_path, nrow=1, padding=0)

                    print(f"Saved evaluation image at {sample_path}")

        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches

        with torch.no_grad():
            similarities = []
            for j in range(min(4, neutral_img.shape[0])):
                x_real_img = neutral_img[j].unsqueeze(0)
                x_fake_img = G_XtoY(x_real_img)
                similarity = calculate_image_similarity(x_real_img[0], x_fake_img[0])
                similarities.append(similarity)

            avg_similarity = np.mean(similarities)

            writer.add_scalar('Metrics/Avg_D_Loss', avg_d_loss, epoch)
            writer.add_scalar('Metrics/Avg_G_Loss', avg_g_loss, epoch)
            writer.add_scalar('Metrics/Image_Similarity', avg_similarity, epoch)

            current_score = avg_g_loss

            if current_score < best_score:
                best_score = current_score
                save_checkpoint(os.path.join(out_path, 'cyclegan_best.pt'), epoch, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, best_score)
                print(f"New best model saved with score: {best_score:.4f}")

        save_checkpoint(os.path.join(out_path, 'cyclegan_last.pt'), epoch, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, best_score)

        print(f"Epoch {epoch+1} completed - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}, Similarity: {avg_similarity:.4f}")

    writer.close()
    print(f"Training completed. Best score: {best_score:.4f}")

if __name__ == '__main__':
    train()
