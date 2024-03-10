import os
import sys

sys.path.append("src/")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb
from piq import psnr, ssim
from tqdm import trange

from models.unet import Unet
from utils.data_utils import FFT, IFFT
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

net = Unet(
    in_chans=in_channels,
    out_chans=out_channels,
    chans=net_channels,
    num_pool_layers=num_pool_layers,
    batchnorm=norm,
    drop_prob=dropout,
).cuda()

learning_rate = 3e-4
EPX = 200

optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPX)


def loss_function(output, target):
    from pytorch_msssim import ssim

    l1_loss = (
        normalize_image(output) - normalize_image(target)
    ).abs().sum() / torch.numel(output)
    return (1 - 0.84) * l1_loss + 0.84 * (
        1 - ssim(target, output, size_average=True, nonnegative_ssim=True)
    )


# load data
data_train = torch.load("Dataset/ClassificationData/train/images/images.pth")
# get the values of the dictionary and save in one tensor
train_batch = []
for key, value in data_train.items():
    if value.shape != (320, 320):
        continue
    train_batch.append(torch.from_numpy(value))
data_train = torch.stack(train_batch, dim=0)

batch_train, H, W = data_train.shape

channels = 1
bsz = 1
bsz_val = 1


config = {
    "learning_rate": learning_rate,
    "batch_size": bsz,
    "batch_size_val": bsz_val,
    "epochs": EPX,
    "Net": {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "channels": net_channels,
        "num_pool_layers": num_pool_layers,
        "norm": norm,
        "dropout": dropout,
    },
}

wandb.init(
    project="MIDL24_MoCo",
    group="ArtRefine_Bboxes",
    name="ArtRefine_bboxes",
    mode="online",
    config=config,
)

for i in range(EPX):

    net.train()
    shuffeled_idx = torch.randperm(batch_train).view(-1, bsz)

    losses = []
    ssims = []
    psnrs = []

    print("Epoch: ", i)
    for batch_idx in trange(batch_train):
        img_train = IFFT(
            FFT(data_train[shuffeled_idx[batch_idx]].view(bsz, channels, H, W).cuda())
        )
        kspace_motion = torch.zeros_like(img_train).cuda()

        with torch.no_grad():
            for trgt in range(bsz):
                kspace_motion[trgt], _, _, _ = motion_simulation2D(
                    img_train[trgt].squeeze()
                )

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

        ssims.append(
            ssim(
                normalize_image(motion_refined), normalize_image(img_gt), data_range=1.0
            )
        )
        psnrs.append(psnr(normalize_image(motion_refined), normalize_image(img_gt)))
        losses.append(loss)

    wandb.log({"loss_train": torch.stack(losses).mean(), "epoch": i})
    wandb.log({"ssim_train": torch.stack(ssims).mean(), "epoch": i})
    wandb.log({"psnr_train": torch.stack(psnrs).mean(), "epoch": i})

torch.save(net.state_dict(), "src/model_weights/unet_denoising_detection_task.pth")
plt.plot(losses.cpu().detach().numpy())
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training loss")
os.makedirs("results/unet_classification", exist_ok=True)
plt.savefig("results/unet_classification/unet_brain_t2_loss.png", dpi=300)
