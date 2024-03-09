import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import sys

sys.path.append('src/')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from piq import SSIMLoss, psnr, ssim
from tqdm import trange

import wandb
from models.unet import Unet
from utils.data_utils import IFFT
from utils.motion_utils import motion_simulation2D


def normalize_image(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def reset_determinism(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

reset_determinism(128)

in_channels = 1
out_channels = 1
net_channels = 32
num_pool_layers = 6
norm = torch.nn.InstanceNorm2d
dropout = 0.0

net = Unet(in_chans=in_channels, out_chans=out_channels, chans=net_channels, num_pool_layers=num_pool_layers, batchnorm=norm, drop_prob=dropout).cuda()

learning_rate = 3e-4
EPX = 200

optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPX)

def loss_function(output, target):
    from pytorch_msssim import ssim
    l1_loss = (normalize_image(output) - normalize_image(target)).abs().sum() / torch.numel(output)
    return (1 - 0.84) * l1_loss + 0.84 * (1 - ssim(target, output,
                                              size_average=True,
                                              nonnegative_ssim=True))
# load data
data_train = torch.load('Dataset/Brain/t2/train_files/_train_data.pth')
data_val = torch.load('Dataset/Brain/t2/val_files/_val_data.pth')

k_space_train = data_train['kspace']
k_space_val = data_val['kspace']

batch_train, H, W = k_space_train.shape
batch_val,_,_ = k_space_val.shape

channels = 1
bsz = 1
bsz_val = 1

best_ssim = 80.0
best_ssim_diff = 10.0

config = {'learning_rate': learning_rate, 'batch_size': bsz, 'batch_size_val': bsz_val, 'epochs': EPX, 'Net': {'in_channels': in_channels, 'out_channels': out_channels, 'channels': net_channels, 'num_pool_layers': num_pool_layers, 'norm': norm, 'dropout': dropout}}

wandb.init(project="MIDL24_MoCo", group="ArtRefine", name="ArtRefine", mode="online", config=config)


for i in trange(EPX):
    
    
    net.train()
    shuffeled_idx = torch.randperm(batch_train).view(-1, bsz)
    
    ssim_train = []
    psnr_train = []
    loss_train = []
    
    for batch_idx in range(batch_train):
        img_train = IFFT(k_space_train[shuffeled_idx[batch_idx]]).view(bsz,channels,H,W).cuda()
        
        kspace_motion = torch.zeros_like(img_train).cuda()
        
        with torch.no_grad():
            for trgt in range(bsz):
                kspace_motion[trgt], _, _, _ = motion_simulation2D(img_train[trgt].squeeze())
        
        img_motion = IFFT(kspace_motion).abs()
        img_gt = img_train.abs()
        # scale per instance
        scale = img_motion.clone().detach().abs().std()
        img_motion = img_motion / scale
        img_gt = img_gt / scale
        
        motion_refined = net(img_motion)

        loss = loss_function(motion_refined, img_gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # collect stats
        ssim_train.append(ssim(normalize_image(motion_refined), normalize_image(img_gt), data_range=1.0))
        psnr_train.append(psnr(normalize_image(motion_refined), normalize_image(img_gt)))
        loss_train.append(loss)

    wandb.log({"loss_train": torch.stack(loss_train).mean(), "epoch": i})
    wandb.log({"ssim_train": torch.stack(ssim_train).mean(), "epoch": i})
    wandb.log({"psnr_train": torch.stack(psnr_train).mean(), "epoch": i})

    # validation    
    net.eval()
    idx_val = torch.randperm(batch_val)[0:bsz_val]
    with torch.no_grad():
        img_val = IFFT(k_space_val[idx_val]).view(bsz_val,channels,H,W).cuda()
        
        kspace_motion_val = torch.zeros_like(img_val).cuda()
        
        # motion simulation on the fly
        for trgt in range(bsz_val):
            kspace_motion_val[trgt], _, _, _ = motion_simulation2D(img_val[trgt].squeeze())
        
        img_motion_val = IFFT(kspace_motion_val).abs()
        
        scale = img_motion_val.detach().std()
        img_motion_val = img_motion_val / scale
        
        img_val_gt = img_val.abs() / scale
        
        motion_refined_val = net(img_motion_val) 
    
        loss_val = loss_function(motion_refined_val, img_val_gt)
        
        ssim_val = ssim(normalize_image(motion_refined_val), normalize_image(img_val_gt), data_range=1.0)
        psnr_val = psnr(normalize_image(motion_refined_val), normalize_image(img_val_gt))
        
        wandb.log({"ssim_val": ssim_val, "epoch": i})
        wandb.log({"psnr_val": psnr_val, "epoch": i})
        
        print(f'Test loss: {loss_val}, SSIM: {ssim_val}, Epoch: {i}')
        
        sim_diff = (ssim_val - ssim(normalize_image(img_motion_val), normalize_image(img_val_gt), data_range=1.0)).abs()
        
        if(sim_diff >= best_ssim_diff):
            best_ssim = ssim_val
            best_ssim_diff = sim_diff
            torch.save(net.state_dict(), f'src/model_weights/unet_denoising_best.pth')
            print('Current best SSIM: ', best_ssim)
            
torch.save(net.state_dict(), f'src/model_weights/unet_denoising_last.pth')          
# plot and example output from the last epoch
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 3, figsize=(15, 10))
axs = axs.ravel()

# font size to 20
plt.rcParams.update({'font.size': 20})
# plot ssim and psnr on the the motion refined and the motion corrupted images

axs[0].imshow(img_motion_val[0].squeeze().detach().cpu().numpy(), cmap='gray')
axs[0].set_axis_off()
axs[0].set_title('Motion corrupted')

ssim_motion = ssim(normalize_image(img_motion_val), normalize_image(img_val_gt), data_range=1.0)
psnr_motion = psnr(normalize_image(img_motion_val), normalize_image(img_val_gt))

# text in the left and right aupper corner and color yellow
axs[0].text(0.5, 0.5, f'SSIM: {ssim_motion:.3f}\nPSNR: {psnr_motion:.3f}', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes, fontsize=20, color='yellow')

axs[1].imshow(motion_refined_val[0].squeeze().detach().cpu().numpy(), cmap='gray')
axs[1].set_axis_off()
axs[1].set_title('Motion refined')

ssim_corr = ssim(normalize_image(motion_refined_val), normalize_image(img_val_gt), data_range=1.0)
psnr_corr = psnr(normalize_image(motion_refined_val), normalize_image(img_val_gt))

# text in the left and right aupper corner and color yellow
axs[1].text(0.5, 0.5, f'SSIM: {ssim_corr:.3f}\nPSNR: {psnr_corr:.3f}', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=20, color='yellow')

axs[2].imshow(img_val[0].abs().squeeze().detach().cpu().numpy(), cmap='gray')
axs[2].set_axis_off()
axs[2].set_title('Ground truth')

plt.tight_layout()
fig.savefig('results/unet_brain_t2_example.png', dpi=300)
plt.close(fig)






    
        
                