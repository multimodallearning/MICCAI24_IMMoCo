import sys

sys.path.append('src/')

import math
import random

import numpy as np
import piq
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.fft import fftn, fftshift, ifftn, ifftshift
from tqdm import tqdm

from utils.evaluate import my_psnr, ssim
from utils.motion_utils import motion_simulation2D
from utils.pytorch_nufft import nufft

Ft = lambda x : fftshift(fftn(ifftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
IFt = lambda x : ifftshift(ifftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

random.seed(228)
torch.manual_seed(228)
torch.cuda.manual_seed(228)
np.random.seed(228)

def normalize(x):
    return ((x - x.min()) / (x.max() - x.min() + 1e-12) )
    
def calc_metrics(y_pred: torch.Tensor, y_gt: torch.Tensor):
    metrics_dict = {}
    metrics_dict['psnr'] = my_psnr(normalize(y_pred), normalize(y_gt)).item()
    metrics_dict['ssim'] = ssim(normalize(y_pred), normalize(y_gt)).item()
    metrics_dict['l1_loss'] = F.l1_loss(y_pred, y_gt).item()
    metrics_dict['ms_ssim'] = piq.multi_scale_ssim(normalize(y_pred),
                                                   normalize(y_gt),
                                                   data_range=1.).item()
    metrics_dict['vif_p'] = piq.vif_p(normalize(y_pred),
                                      normalize(y_gt), data_range=1.).item()
    return metrics_dict


# Algorithm
def get_rot_mat_nufft(rot_vector):
    rot_mat = torch.zeros(rot_vector.shape[0], 2, 2).cuda()
    rot_mat[:, 0, 0] = torch.cos(rot_vector)
    rot_mat[:, 0, 1] = -torch.sin(rot_vector)
    rot_mat[:, 1, 0] = torch.sin(rot_vector)
    rot_mat[:, 1, 1] = torch.cos(rot_vector)
    return rot_mat


def R_differentiable(ks, rot_vector, oversamp=5):
    rot_matrices = get_rot_mat_nufft(rot_vector)
    grid = torch.stack([
        arr.flatten() for arr in torch.meshgrid(
            torch.arange(-ks.shape[0]//2, ks.shape[0]//2).float(),
            torch.arange(-ks.shape[1]//2, ks.shape[1]//2).float(),
            indexing='ij')]).cuda()
    grid = (rot_matrices @ \
            grid.reshape(2, 320, 320).movedim(1, 0)).movedim(0, 1).reshape(2, -1)
    img = nufft.nufft_adjoint(ks, grid.T, device='cuda', oversamp=oversamp, 
                              out_shape=[1, 1, *ks.shape])[0, 0]
    return Ft(img)


def loss_function(output, target):
    from pytorch_msssim import ssim
    l1_loss = (normalize(output) - normalize(target)).abs().sum() / torch.numel(output)
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

# configs
learning_rate = 5e-5
EPX = 200
in_channels = 1
out_channels = 1
net_channels = 32
num_pool_layers = 6
norm = torch.nn.InstanceNorm2d
dropout = 0.0
bsz = 1
bsz_val = 1

# Run Algorithm with U-Net 
#unet = Unet(1, 1, 32, 6, batchnorm=torch.nn.InstanceNorm2d, init_type=args.init).cuda()
# load model
from models.unet import Unet

unet = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=6, batchnorm=nn.InstanceNorm2d ,drop_prob=0.0).cuda()

optimizer_unet = torch.optim.Adam(unet.parameters(), lr=5e-5, betas=(0.9, 0.999)) 
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=EPX)

config = {'learning_rate': learning_rate, 'batch_size': bsz, 'batch_size_val': bsz_val, 'epochs': EPX, 'Net': {'in_channels': in_channels, 'out_channels': out_channels, 'channels': net_channels, 'num_pool_layers': num_pool_layers, 'norm': norm, 'dropout': dropout}}

wandb.init(project="MICCAI24_MoCo", group="AFPLUS", name="AFPLUS", mode="online", config=config)

beta1, beta2 = 0.89, 0.8999

metric_buf = {'psnr': 20.0,
                'ssim': 0.4}

for epoch in range(EPX):
    print('-'*20, 'For Epoch: ', epoch, '-'*20)
    # Training
    losses_train = []
    train_metrics = []
    
    unet.train()
    
    # Shuffle idxs of Data Samples
    shuff_idx = np.arange(200)
    np.random.shuffle(shuff_idx)
    
    pbar = tqdm(enumerate(shuff_idx), total=50)
    for i, batch_idx in pbar:
        gt_ks = k_space_train[batch_idx]
        img_train = IFt(gt_ks).cuda()
        ks = torch.zeros_like(img_train).cuda()
        with torch.no_grad():
            #for trgt in range(1):
            ks, mask, _, _ = motion_simulation2D(img_train)
            ks = Ft((IFt(ks) - IFt(ks).abs().mean()) / (IFt(ks).abs().std() + 1e-11))
             
        ps = ks.shape[-1]
        ps_cf = int(ps//2 * 0.08)
        zero_middle = torch.ones((ps)).cuda()
        zero_middle[ps//2 - ps_cf:ps//2 + ps_cf] = 0.
        img = IFt(ks).abs()
        gt_img = IFt(gt_ks).abs()
        x_shifts = torch.zeros(ps)
        y_shifts = torch.zeros(ps)
        x_shifts = torch.nn.Parameter(data=x_shifts.cuda(), requires_grad=True)
        y_shifts = torch.nn.Parameter(data=y_shifts.cuda(), requires_grad=True)
        x_moment1 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
        x_moment2 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
        y_moment1 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
        y_moment2 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
        rot_vector = torch.zeros(ps).cuda()
        rot_vector = torch.nn.Parameter(data=rot_vector.cuda(), requires_grad=True)
        rot_moment1 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
        rot_moment2 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
        for _ in range(30):
            # rot_vector = rot_vector * zero_middle
            # x_shifts = x_shifts * zero_middle
            # y_shifts = y_shifts * zero_middle

            # Translation
            phase_shift = -2 * math.pi * (
                x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
                y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
            new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + \
                                                    phase_shift)).exp()
            # Rotation
            yp_ks = R_differentiable(new_k_space, rot_vector)
            yp_img = IFt(yp_ks).abs()
    
            loss_net = (yp_img[None, None] * 1e4 * \
                        unet(yp_img[None, None] * 1e4).sigmoid()).mean()

            x_grad, y_grad, rot_grad = torch.autograd.grad(loss_net, [x_shifts, y_shifts, rot_vector],
                                                            create_graph=True)
            x_grad, y_grad, rot_grad = x_grad * 1e-4, y_grad * 1e-4, rot_grad * 1e-4
            x_moment1 = beta1 * x_moment1.detach() + (1. - beta1) * x_grad
            x_moment2 = beta2 * x_moment2.detach() + (1. - beta2) * x_grad * x_grad + 1e-24
            y_moment1 = beta1 * y_moment1.detach() + (1. - beta1) * y_grad
            y_moment2 = beta2 * y_moment2.detach() + (1. - beta2) * y_grad * y_grad + 1e-24
            rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
            rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
            x_shifts = x_shifts - 3e-4 * x_moment1 * x_moment2.rsqrt()
            y_shifts = y_shifts - 3e-4 * y_moment1 * y_moment2.rsqrt()
            rot_vector = rot_vector - 3e-4  * rot_moment1 * rot_moment2.rsqrt()
            x_shifts.retain_grad()
            y_shifts.retain_grad()
            rot_vector.retain_grad()
            
        # rot_vector = rot_vector * zero_middle
        # x_shifts = x_shifts * zero_middle
        # y_shifts = y_shifts * zero_middle
        # Translation
        phase_shift = -2 * math.pi * (
            x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
            y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
        new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + \
                                                phase_shift)).exp()
        # Rotation
        yp_ks = R_differentiable(new_k_space, rot_vector)

        loss_img = loss_function(IFt(yp_ks.cuda()).abs()[None, None] * 1e4,
                                IFt(gt_ks.cuda()).abs()[None, None] * 1e4)
        
        losses_train.append(loss_img.cpu().item())
        loss_img.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        if i % 16 == 0 and i != 0:
            optimizer_unet.step()
            lr_scheduler.step()
            optimizer_unet.zero_grad()
            
        losses_train.append(loss_img.cpu().item())
        train_metrics.append(calc_metrics(IFt(yp_ks).abs().data.cpu()[None, None],
                                        IFt(gt_ks).abs().data.cpu()[None, None]))
        pbar.set_description('loss: {:.4}'.format(loss_img.item()))
        
    losses_train = np.array(losses_train)
    ssim_vals_train = np.array([d['ssim'] for d in train_metrics])
    psnr_vals_train = np.array([d['psnr'] for d in train_metrics])
    vif_vals_train = np.array([d['vif_p'] for d in train_metrics])
    ms_ssim_vals_train = np.array([d['ms_ssim'] for d in train_metrics])
    l1_loss_vals_train = np.array([d['l1_loss'] for d in train_metrics])
    
    wandb.log({'Train Loss': losses_train.mean(),
                'Train SSIM': ssim_vals_train.mean(),
                 'Train PSNR': psnr_vals_train.mean(),
                 'Train VIF': vif_vals_train.mean(),
                 'Train MS-SSIM': ms_ssim_vals_train.mean(),
                 'Train L1 Loss': l1_loss_vals_train.mean()})
    
    # Validation
    if epoch % 5 == 0: 
        unet.eval()
        new_metrics = []
        losses_val = []
        idx = 0

        for batch in tqdm(range(1)):
            gt_ks = k_space_train[batch_idx]
            img_train = IFt(gt_ks).cuda()
            
            ks = torch.zeros_like(img_train).cuda()
            with torch.no_grad():
                #for trgt in range(1):
                ks, mask, _, _ = motion_simulation2D(img_train)
                ks = Ft((IFt(ks) - IFt(ks).abs().mean()) / (IFt(ks).abs().std() + 1e-11))

            ps = ks.shape[-1]
            ps_cf = int(ps//2 * 0.08)
            zero_middle = torch.ones((ps)).cuda()
            zero_middle[ps//2 - ps_cf:ps//2 + ps_cf] = 0.
            img, gt_img = IFt(ks).abs(), IFt(gt_ks).abs()

            x_shifts = torch.zeros(ps)
            y_shifts = torch.zeros(ps)
            x_shifts = torch.nn.Parameter(data=x_shifts.cuda(), requires_grad=True)
            y_shifts = torch.nn.Parameter(data=y_shifts.cuda(), requires_grad=True)
            x_moment1 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
            x_moment2 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
            y_moment1 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
            y_moment2 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
            rot_vector = torch.zeros(ps).cuda()
            rot_vector = torch.nn.Parameter(data=rot_vector.cuda(), requires_grad=True)
            rot_moment1 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
            rot_moment2 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)

            for _ in range(80):
                # rot_vector = rot_vector * zero_middle
                # x_shifts = x_shifts * zero_middle
                # y_shifts = y_shifts * zero_middle

                # Translation
                phase_shift = -2 * math.pi * (
                    x_shifts * torch.linspace(0, 320, 320)[None, :, None].cuda() + 
                    y_shifts * torch.linspace(0, 320, 320)[None, None, :].cuda())[0]
                new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
                # Rotation
                yp_ks = R_differentiable(new_k_space, rot_vector)
                yp_img = IFt(yp_ks).abs()

                loss_net = (yp_img[None, None] * 1e4 * unet(yp_img[None, None] * 1e4).sigmoid()).mean()
                x_grad, y_grad, rot_grad = torch.autograd.grad(loss_net,
                                                                [x_shifts, y_shifts, rot_vector],
                                                                create_graph=False)
                x_grad, y_grad = x_grad * 1e-4, y_grad * 1e-4
                rot_grad = rot_grad * 1e-4
                x_moment1 = beta1 * x_moment1.detach() + (1. - beta1) * x_grad
                x_moment2 = beta2 * x_moment2.detach() + (1. - beta2) * x_grad * x_grad + 1e-24
                y_moment1 = beta1 * y_moment1.detach() + (1. - beta1) * y_grad
                y_moment2 = beta2 * y_moment2.detach() + (1. - beta2) * y_grad * y_grad + 1e-24
                rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
                rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
                x_shifts = x_shifts - 3e-4 * x_moment1 * x_moment2.rsqrt()
                y_shifts = y_shifts - 3e-4 * y_moment1 * y_moment2.rsqrt()
                rot_vector = rot_vector - 3e-4 * rot_moment1 * rot_moment2.rsqrt()

            # rot_vector = rot_vector * zero_middle
            # x_shifts = x_shifts * zero_middle
            # y_shifts = y_shifts * zero_middle

            # Translation
            phase_shift = -2 * math.pi * (
                x_shifts * torch.linspace(0, 320, 320)[None, :, None].cuda() + 
                y_shifts * torch.linspace(0, 320, 320)[None, None, :].cuda())[0]
            new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + phase_shift)).exp()
            # Rotation
            yp_ks = R_differentiable(new_k_space, rot_vector)

            loss_img = loss_function(IFt(yp_ks.cuda()).abs()[None, None] * 1e4,
                                    IFt(gt_ks.cuda()).abs()[None, None] * 1e4)
            losses_val.append(loss_img.cpu().item())
            new_metrics.append(calc_metrics(IFt(yp_ks).abs().data.cpu()[None, None],
                                            IFt(gt_ks).abs().data.cpu()[None, None]))
               
            idx += 1

        losses_val = np.array(losses_val)
        ssim_vals = np.array([d['ssim'] for d in new_metrics])
        psnr_vals = np.array([d['psnr'] for d in new_metrics])
        vif_vals = np.array([d['vif_p'] for d in new_metrics])
        ms_ssim_vals = np.array([d['ms_ssim'] for d in new_metrics])
        l1_loss_vals = np.array([d['l1_loss'] for d in new_metrics])

        wandb.log({ 'Val Loss': losses_val.mean(),
                    'Val SSIM': ssim_vals.mean(),
                    'Val PSNR': psnr_vals.mean(),
                    'Val VIF': vif_vals.mean(),
                    'Val MS-SSIM': ms_ssim_vals.mean(),
                    'Val L1 Loss': l1_loss_vals.mean()})
        
        # log validation images
        img_batch = np.zeros((3, 320, 320))  # normalize [0,1]
        img_batch[0] = normalize(IFt(ks).abs().cpu().detach()).numpy()[None]
        img_batch[1] = normalize(IFt(yp_ks).abs().cpu().detach()).numpy()[None]
        img_batch[2] = normalize(IFt(gt_ks).abs().cpu().detach()).numpy()[None]
        wandb.log({"Val Images0": [wandb.Image(img_batch[0], caption="Corrupted Image")]})
        wandb.log({"Val Images1": [wandb.Image(img_batch[1], caption="Reconstructed Image")]})
        wandb.log({"Val Images2": [wandb.Image(img_batch[2], caption="Ground Truth Image")]})
        
        # Save Model Weights
        if ssim_vals.mean() > metric_buf['ssim'] and psnr_vals.mean() > metric_buf['psnr']:
            metric_buf['ssim'] = ssim_vals.mean()
            metric_buf['psnr'] = psnr_vals.mean()
            torch.save(unet.state_dict(),  'src/model_weights/AFPlus_best.pth')

torch.save(unet.state_dict(), 'src/model_weights/AFPlus.pth')
