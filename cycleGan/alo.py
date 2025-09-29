# import os
# import shutil
#
# base_dir = r"C:\Users\tam\Documents\data\monet_style_dataset"
#
# # nơi chứa thư mục gom
# origin_dir = os.path.join(base_dir, "origin")
# style_dir = os.path.join(base_dir, "style")
#
# # tạo thư mục đích
# os.makedirs(origin_dir, exist_ok=True)
# os.makedirs(style_dir, exist_ok=True)
#
# # ánh xạ: (thư mục gốc -> thư mục đích)
# mapping = {
#     "monet_style_dataset_A/original_A": origin_dir,
#     "monet_style_dataset_B/original_B": origin_dir,
#     "monet_style_dataset_A/stylized_A": style_dir,
#     "monet_style_dataset_B/stylized_B": style_dir,
# }
#
# for src_rel, dst in mapping.items():
#     src = os.path.join(base_dir, src_rel)
#     os.makedirs(dst, exist_ok=True)
#
#     for fname in os.listdir(src):
#         fsrc = os.path.join(src, fname)
#         fdst = os.path.join(dst, fname)
#         if os.path.isfile(fsrc):
#             # nếu trùng tên thì thêm prefix A_ hoặc B_
#             prefix = "A_" if "_A" in src_rel else "B_"
#             fdst = os.path.join(dst, prefix + fname)
#             shutil.move(fsrc, fdst)


import os
import shutil
import random

base_dir = r"C:\Users\tam\Documents\data\monet_style_dataset"
origin_dir = os.path.join(base_dir, "origin")
style_dir = os.path.join(base_dir, "style")

# thư mục train/val
train_origin = os.path.join(base_dir, "train", "origin")
train_style = os.path.join(base_dir, "train", "style")
val_origin = os.path.join(base_dir, "val", "origin")
val_style = os.path.join(base_dir, "val", "style")

for d in [train_origin, train_style, val_origin, val_style]:
    os.makedirs(d, exist_ok=True)

# giả sử file origin và style trùng tên (A_xxx.jpg ↔ A_xxx.jpg)
files = sorted(os.listdir(origin_dir))
random.shuffle(files)

split_idx = int(len(files) * 0.9)
train_files = files[:split_idx]
val_files = files[split_idx:]

def move_pairs(file_list, dst_origin, dst_style):
    for fname in file_list:
        f_origin = os.path.join(origin_dir, fname)
        f_style = os.path.join(style_dir, fname)
        if os.path.exists(f_origin) and os.path.exists(f_style):
            shutil.copy2(f_origin, os.path.join(dst_origin, fname))
            shutil.copy2(f_style, os.path.join(dst_style, fname))

# chia tập
move_pairs(train_files, train_origin, train_style)
move_pairs(val_files, val_origin, val_style)

# def save_checkpoint(filepath, epoch, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, best_score=None):
#     checkpoint = {
#         'epoch': epoch,
#         'G_XtoY_state_dict': G_XtoY.state_dict(),
#         'G_YtoX_state_dict': G_YtoX.state_dict(),
#         'D_X_state_dict': D_X.state_dict(),
#         'D_Y_state_dict': D_Y.state_dict(),
#         'g_optimizer_state_dict': g_optimizer.state_dict(),
#         'd_optimizer_state_dict': d_optimizer.state_dict(),
#         'best_score': best_score
#     }
#     torch.save(checkpoint, filepath)
#
# def load_checkpoint(filepath, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, device):
#     if os.path.exists(filepath):
#         checkpoint = torch.load(filepath, map_location=device)
#         G_XtoY.load_state_dict(checkpoint['G_XtoY_state_dict'])
#         G_YtoX.load_state_dict(checkpoint['G_YtoX_state_dict'])
#         D_X.load_state_dict(checkpoint['D_X_state_dict'])
#         D_Y.load_state_dict(checkpoint['D_Y_state_dict'])
#         g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
#         d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
#         best_score = checkpoint.get('best_score', float('inf'))
#         return checkpoint['epoch'], best_score
#     return 0, float('inf')
#
# def train():
#     batch_size = 4
#     lr = 1e-4
#     num_epochs = 100
#     image_size = 256
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#     out_path = "out1"
#     os.makedirs(out_path, exist_ok=True)
#
#     transform = Compose([
#         Resize((image_size, image_size)),
#         ToTensor(),
#         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
#
#     train_dataset = Monet(root="C:/Users/tam/Documents/data/monet_style_dataset", is_train=True, transform=transform)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
#
#     val_dataset = Monet(root="C:/Users/tam/Documents/data/monet_style_dataset", is_train=False, transform=transform)
#     val_dataloader = DataLoader(
#         dataset=val_dataset,
#         batch_size=batch_size,
#         num_workers=8,
#         shuffle=True,
#         drop_last=True
#     )
#
#     G_XtoY = Generator().to(device)
#     G_YtoX = Generator().to(device)
#     D_X = Discriminator().to(device)
#     D_Y = Discriminator().to(device)
#
#     g_optimizer = torch.optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=lr, betas=(0.5, 0.99))
#     d_optimizer = torch.optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()), lr=lr, betas=(0.5, 0.99))
#
#     adversarial_loss = nn.BCEWithLogitsLoss()
#     cycle_loss = nn.L1Loss()
#
#     start_epoch, best_score = load_checkpoint(os.path.join(out_path, 'cyclegan_checkpoint.pt'), G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, device)
#
#     writer = SummaryWriter(os.path.join(out_path, 'tensorboard_logs'))
#
#     neutral_img, happy_img = next(iter(val_dataloader))
#     neutral_img = neutral_img.to(device)
#     happy_img = happy_img.to(device)
#
#     def calculate_image_similarity(img1, img2):
#         # Simple MSE-based similarity (lower is better, so we return 1-MSE for higher=better)
#         mse = F.mse_loss(img1, img2).item()
#         return 1.0 / (1.0 + mse)  # Convert to similarity score (0-1, higher is better)
#
#     for epoch in range(start_epoch, num_epochs):
#         epoch_d_loss = 0
#         epoch_g_loss = 0
#         num_batches = 0
#
#         progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
#         for i, (x_real, y_real) in enumerate(progress_bar):
#             x_real, y_real = x_real.to(device), y_real.to(device)
#
#             # Train D
#             y_fake = G_XtoY(x_real)
#             x_fake = G_YtoX(y_real)
#
#             d_loss_x = adversarial_loss(D_X(x_real), torch.ones_like(D_X(x_real))) + \
#                        adversarial_loss(D_X(x_fake.detach()), torch.zeros_like(D_X(x_fake)))
#             d_loss_y = adversarial_loss(D_Y(y_real), torch.ones_like(D_Y(y_real))) + \
#                        adversarial_loss(D_Y(y_fake.detach()), torch.zeros_like(D_Y(y_fake)))
#
#             d_loss = (d_loss_x + d_loss_y) / 2
#             d_optimizer.zero_grad()
#             d_loss.backward()
#             d_optimizer.step()
#
#             # Train G
#             y_fake = G_XtoY(x_real)
#             x_fake = G_YtoX(y_real)
#
#             x_recon = G_YtoX(y_fake)
#             y_recon = G_XtoY(x_fake)
#             g_loss_x = adversarial_loss(D_X(x_fake), torch.ones_like(D_X(x_fake)))
#             g_loss_y = adversarial_loss(D_Y(y_fake), torch.ones_like(D_Y(y_fake)))
#
#             cycle_loss_x = cycle_loss(x_recon, x_real)
#             cycle_loss_y = cycle_loss(y_recon, y_real)
#
#             g_loss = g_loss_x + g_loss_y + 10 * (cycle_loss_x + cycle_loss_y)
#             g_optimizer.zero_grad()
#             g_loss.backward()
#             g_optimizer.step()
#
#             epoch_d_loss += d_loss.item()
#             epoch_g_loss += g_loss.item()
#             num_batches += 1
#
#             progress_bar.set_postfix({
#                 'D_Loss': f'{d_loss.item():.4f}',
#                 'G_Loss': f'{g_loss.item():.4f}'
#             })
#
#             global_step = epoch * len(train_dataloader) + i
#             if i % 50 == 0:
#                 writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
#                 writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
#                 writer.add_scalar('Loss/Cycle_X', cycle_loss_x.item(), global_step)
#                 writer.add_scalar('Loss/Cycle_Y', cycle_loss_y.item(), global_step)
#
#             # Evaluation step
#             if (i % 100 == 0) and (i != 0):
#                 with torch.no_grad():
#                     paired_images = []
#
#                     for j in range(neutral_img.shape[0]):
#                         x_real_img = neutral_img[j].unsqueeze(0)
#                         x_fake_img = G_XtoY(x_real_img)
#                         paired_images.append(torch.cat([x_real_img, x_fake_img], dim=3))
#                     for j in range(happy_img.shape[0]):
#                         y_real_img = happy_img[j].unsqueeze(0)
#                         y_fake_img = G_YtoX(y_real_img)
#                         paired_images.append(torch.cat([y_real_img, y_fake_img], dim=3))
#
#                     x_concat = torch.cat(paired_images, dim=2)
#
#                     x_concat = ((x_concat + 1) / 2).clamp_(0, 1)
#
#                     sample_path = os.path.join(out_path, f'epoch{epoch}_step{i}-images.jpg')
#                     save_image(x_concat, sample_path, nrow=1, padding=0)
#
#                     print(f"Saved evaluation image at {sample_path}")
#
#         avg_d_loss = epoch_d_loss / num_batches
#         avg_g_loss = epoch_g_loss / num_batches
#
#         with torch.no_grad():
#             similarities = []
#             for j in range(min(4, neutral_img.shape[0])):
#                 x_real_img = neutral_img[j].unsqueeze(0)
#                 x_fake_img = G_XtoY(x_real_img)
#                 similarity = calculate_image_similarity(x_real_img[0], x_fake_img[0])
#                 similarities.append(similarity)
#
#             avg_similarity = np.mean(similarities)
#
#             writer.add_scalar('Metrics/Avg_D_Loss', avg_d_loss, epoch)
#             writer.add_scalar('Metrics/Avg_G_Loss', avg_g_loss, epoch)
#             writer.add_scalar('Metrics/Image_Similarity', avg_similarity, epoch)
#
#             current_score = avg_g_loss
#
#             if current_score < best_score:
#                 best_score = current_score
#                 save_checkpoint(os.path.join(out_path, 'cyclegan_best.pt'), epoch, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, best_score)
#                 print(f"New best model saved with score: {best_score:.4f}")
#
#         save_checkpoint(os.path.join(out_path, 'cyclegan_last.pt'), epoch, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, best_score)
#
#         print(f"Epoch {epoch+1} completed - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}, Similarity: {avg_similarity:.4f}")
#
#     writer.close()
#     print(f"Training completed. Best score: {best_score:.4f}")
