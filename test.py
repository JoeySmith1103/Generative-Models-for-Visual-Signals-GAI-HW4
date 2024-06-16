import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToPILImage

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_subset = Subset(train_dataset, list(range(1)))
train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)

## DDPM
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4)
).cuda()
diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
).cuda()


# DIP model
class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def generate_noisy_images(diffusion, image, noise_levels):
    noisy_images = []
    for noise_level in noise_levels:
        t = torch.tensor([int(noise_level * (diffusion.num_timesteps - 1))], device=image.device).long()
        noisy_image = diffusion.q_sample(image, t)
        noisy_images.append(noisy_image)
    return noisy_images


def train_dip_model_with_ddpm(dip_model, diffusion, dataloader, num_epochs, noise_levels):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dip_model.parameters(), lr=0.001)
    best_psnr = 0
    best_model = None
    psnr_values = []
    ssim_values = []
    loss_values = []

    for epoch in range(num_epochs):
        total_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        for batch in dataloader:
            images, _ = batch
            noisy_images = generate_noisy_images(diffusion, images.cuda(), noise_levels)
            for noisy_image in noisy_images:
                optimizer.zero_grad()
                outputs = dip_model(noisy_image)
                loss = criterion(outputs, images.cuda())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # calculate PSNR and SSIM
                images_np = images.cpu().numpy()
                outputs_np = outputs.detach().cpu().numpy()
                if images_np.shape[0] > 1:
                    batch_psnr = psnr(images_np, outputs_np, data_range=1.0)
                    batch_ssim = ssim(images_np, outputs_np, multichannel=True, win_size=3, data_range=1.0, channel_axis=-1)
                else:
                    batch_psnr = psnr(images_np[0], outputs_np[0], data_range=1.0)
                    batch_ssim = ssim(images_np[0], outputs_np[0], multichannel=True, win_size=3, data_range=1.0, channel_axis=-1)
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim

        avg_loss = total_loss / (len(dataloader) * len(noise_levels))
        avg_psnr = epoch_psnr / (len(dataloader) * len(noise_levels))
        avg_ssim = epoch_ssim / (len(dataloader) * len(noise_levels))

        psnr_values.append(avg_psnr)
        ssim_values.append(avg_ssim)
        loss_values.append(avg_loss)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_model = dip_model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
        else:
            print(f'Epoch: {epoch + 1}')

    torch.save(best_model, 'best_dip_model_with_ddpm.pth')
    return psnr_values, ssim_values, loss_values


# only with DIP model
def train_traditional_dip_model(dip_model, dataloader, num_epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dip_model.parameters(), lr=0.001)
    best_psnr = 0
    best_model = None
    psnr_values = []
    ssim_values = []
    loss_values = []

    for epoch in range(num_epochs):
        total_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        for batch in dataloader:
            images, _ = batch
            optimizer.zero_grad()
            outputs = dip_model(images.cuda())
            loss = criterion(outputs, images.cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 计算 PSNR 和 SSIM
            images_np = images.cpu().numpy()
            outputs_np = outputs.detach().cpu().numpy()
            if images_np.shape[0] > 1:
                batch_psnr = psnr(images_np, outputs_np, data_range=1.0)
                batch_ssim = ssim(images_np, outputs_np, multichannel=True, win_size=3, data_range=1.0, channel_axis=-1)
            else:
                batch_psnr = psnr(images_np[0], outputs_np[0], data_range=1.0)
                batch_ssim = ssim(images_np[0], outputs_np[0], multichannel=True, win_size=3, data_range=1.0, channel_axis=-1)
            epoch_psnr += batch_psnr
            epoch_ssim += batch_ssim

        avg_loss = total_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)

        psnr_values.append(avg_psnr)
        ssim_values.append(avg_ssim)
        loss_values.append(avg_loss)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_model = dip_model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')

    torch.save(best_model, 'best_traditional_dip_model.pth')
    return psnr_values, ssim_values, loss_values

# only DIP training
traditional_dip_model = DIPModel().cuda()
traditional_psnr_values, traditional_ssim_values, traditional_losses = train_traditional_dip_model(traditional_dip_model, train_loader, num_epochs=500)
# DIP with DDPM training
dip_model = DIPModel().cuda()
psnr_values, ssim_values, new_losses = train_dip_model_with_ddpm(dip_model, diffusion, train_loader, num_epochs=500, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5])


def plot_comparison_results(new_results, traditional_results, new_losses, traditional_losses):
    psnr_improvement = np.array(new_results['psnr']) - np.array(traditional_results['psnr'])
    ssim_improvement = np.array(new_results['ssim']) - np.array(traditional_results['ssim'])

    epochs = range(1, len(psnr_improvement) + 1)

    plt.figure(figsize=(18, 6))

    # PSNR improvement
    plt.subplot(1, 3, 1)
    plt.plot(epochs, psnr_improvement, label='PSNR Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement')
    plt.title('PSNR Improvement Over Traditional DIP')
    plt.legend()

    # SSIM improvement
    plt.subplot(1, 3, 2)
    plt.plot(epochs, ssim_improvement, label='SSIM Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement')
    plt.title('SSIM Improvement Over Traditional DIP')
    plt.legend()

    # Loss comparison
    plt.subplot(1, 3, 3)
    plt.plot(epochs, new_losses, label='DIP with DDPM Loss')
    plt.plot(epochs, traditional_losses, label='Traditional DIP Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

new_results = {
    'psnr': psnr_values,
    'ssim': ssim_values
}
traditional_results = {
    'psnr': traditional_psnr_values,
    'ssim': traditional_ssim_values
}

plot_comparison_results(new_results, traditional_results, new_losses, traditional_losses)


# show result
def save_reconstructed_images(model, dataloader, noise_levels, output_dir='output_images'):
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, _ = batch
            noisy_images = generate_noisy_images(diffusion, images.cuda(), noise_levels)
            for j, noisy_image in enumerate(noisy_images):
                outputs = model(noisy_image)

                for k in range(images.size(0)):
                    original_image = ToPILImage()(images[k].cpu())
                    noisy_image_pil = ToPILImage()(noisy_image[k].cpu())
                    reconstructed_image = ToPILImage()(outputs[k].cpu())

                    original_image.save(os.path.join(output_dir, f'original_{i}_{k}.png'))
                    noisy_image_pil.save(os.path.join(output_dir, f'noisy_{i}_{k}_{j}.png'))
                    reconstructed_image.save(os.path.join(output_dir, f'reconstructed_{i}_{k}_{j}.png'))

                    if k == 0:
                        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                        axs[0].imshow(original_image)
                        axs[0].set_title('Original Image')
                        axs[1].imshow(noisy_image_pil)
                        axs[1].set_title(f'Noisy Image (Noise Level {j})')
                        axs[2].imshow(reconstructed_image)
                        axs[2].set_title('Reconstructed Image')
                        plt.show()


save_reconstructed_images(dip_model, train_loader, [0.1, 0.2, 0.3, 0.4, 0.5])
