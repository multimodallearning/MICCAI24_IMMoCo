import os
import sys

sys.path.append("src/")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from models.autofocusing import Autofocusing
from models.KLineDetect import get_unet
from utils.data_utils import FFT, IFFT, normalize_image
from utils.evaluate import calmetric2D
from utils.motion_utils import extract_movement_groups


class GradientEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def entropy(self, x):
        return -torch.sum(torch.mul(x, torch.log(x + 1e-8)))

    def forward(self, x):

        # loss = 0
        # for x in [x.real, x.imag]:
        dx = (x[:, :-1] - x[:, 1:]).abs()
        dy = ((x[:-1, :] - x[1:, :])).abs()

        # pad the gradient
        dx = F.pad(dx, (0, 1, 0, 0), mode="constant", value=0)
        dy = F.pad(dy, (0, 0, 0, 1), mode="constant", value=0)

        gradient = dx + dy

        loss = self.entropy(gradient)

        return loss


# load model
net = get_unet(
    in_chans=2, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0
).cuda()
net.load_state_dict(torch.load("src/model_weights/KLineDetect_4200_epochs_v2.pth"))

if os.path.exists("results/autofocusing") == False:
    os.mkdir("results/autofocusing")

scenarioius = ["light", "heavy"]
metrics_all = []


for scenario in scenarioius:
    print("Loading data...")
    data_path = "Dataset/Brain/t2/test_files/_test_data_" + scenario + ".pth"
    data_test = torch.load(data_path)

    kspaces_test = data_test["kspace_motion"]
    images_gt = data_test["image_rss"]

    metrics = []
    batch_test, H, W = kspaces_test.shape

    bsz = 1
    channels = 1
    net.eval()

    metrics = []
    for idx_test in trange(batch_test):

        k_space_test = kspaces_test[idx_test].view(bsz, channels, H, W).cuda()
        img_motion_test = IFFT(k_space_test).abs()

        with torch.no_grad():
            mask = (
                net(
                    torch.view_as_real(k_space_test).squeeze(1).permute(0, 3, 1, 2)
                ).sigmoid()
                > 0.5
            ).float()
            masks = extract_movement_groups(
                mask.squeeze().sum(0).div(mask.squeeze().shape[0]) > 0.2, make_list=True
            ).cuda()

        scale = img_motion_test.detach().abs().max()
        k_space_test = k_space_test / scale
        image_gt = images_gt[idx_test].abs().cuda() / scale

        AUTOFOCUS = Autofocusing(masks).cuda()
        optimizer = torch.optim.Adam(AUTOFOCUS.parameters(), lr=1.0)

        for i in range(60):
            optimizer.zero_grad()
            kspace_refined = AUTOFOCUS(k_space_test.squeeze())
            loss = GradientEntropyLoss()(IFFT(kspace_refined)) * 1e-4
            loss.backward()
            optimizer.step()

        motion_refined_test = IFFT(kspace_refined).abs().detach()
        # psnr_array, ssim_array, haar_psi_array, rmse_array
        crop = [int(H / 4), int(W / 4)]
        image_gt_crop = image_gt[crop[0] : -crop[0], crop[1] : -crop[1]]
        motion_refined_test_crop = motion_refined_test[
            crop[0] : -crop[0], crop[1] : -crop[1]
        ]
        psnr_test, ssim_test, haar_psi_test, rmse_test = calmetric2D(
            motion_refined_test_crop.unsqueeze(0).unsqueeze(0),
            image_gt_crop.unsqueeze(0).unsqueeze(0),
        )
        # psnr_test, ssim_test, haar_psi_test, rmse_test = calmetric2D(motion_refined_test.unsqueeze(0).unsqueeze(0), image_gt.unsqueeze(0).unsqueeze(0))

        metrics.append(
            {
                "ssim": ssim_test,
                "psnr": psnr_test,
                "haar_psi": haar_psi_test,
                "rmse": rmse_test,
            }
        )

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].imshow(img_motion_test.squeeze().cpu().numpy(), cmap="gray")
    axs[0].set_title("Motion corrupted image", fontsize=16)
    axs[1].imshow(motion_refined_test.squeeze().cpu().numpy(), cmap="gray")
    axs[1].set_title("Motion corrected image", fontsize=16)
    axs[2].imshow(image_gt.squeeze().cpu().numpy(), cmap="gray")
    axs[2].set_title("Ground truth image", fontsize=16)

    fig.savefig(
        "results/autofocusing/autofocus_" + scenario + ".png", bbox_inches="tight"
    )
    plt.close(fig)

    metrics_all.append(metrics)

# save metrics
torch.save(metrics_all, "results/autofocusing/autofocus_metrics.pth")

# load the metrics and loop over the list of lists of dictionaries, get the means and stds of each scenario for each metric, plot the violin plots and save them as a latex table
metrics_all = torch.load("results/autofocusing/autofocus_metrics.pth")

metrics_for_plot = []
means_all = []
stds_all = []

for scenario in range(len(metrics_all)):

    for metric in metrics_all[scenario][0].keys():

        metrics = []
        for i in range(len(metrics_all[scenario])):
            metrics.append(metrics_all[scenario][i][metric])

        metrics_for_plot.append(torch.tensor(metrics))
        means_all.append(torch.mean(torch.tensor(metrics)))
        stds_all.append(torch.std(torch.tensor(metrics)))

metrics_for_plot = torch.stack(metrics_for_plot).squeeze().view(8, 50)
means_all = torch.stack(means_all)
stds_all = torch.stack(stds_all)

# save means +- stds in a latex table
print("Saving latex table...")
with open("results/autofocusing/autofocus_metrics.tex", "w") as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{autofocus_metrics metrics}\n")
    f.write("\\label{tab:autofocus_metrics}\n")
    f.write("\\begin{tabular}{l|cccc}\n")
    f.write("\\topline\n")
    f.write("Scenario & SSIM & PSNR & Haar psi & RMSE \\\\ \n")
    f.write("\\midline\n")
    for i in range(len(scenarioius)):
        f.write(
            scenarioius[i]
            + " & "
            + f"{means_all[i*4]:.2f}"
            + "$\pm$"
            + f"{stds_all[i*4]:.2f}"
            + " & "
            + f"{means_all[i*4+1]:.2f}"
            + "$\pm$"
            + f"{stds_all[i*4+1]:.2f}"
            + " & "
            + f"{means_all[i*4+2]:.2f}"
            + "$\pm$"
            + f"{stds_all[i*4+2]:.2f}"
            + " & "
            + f"{means_all[i*4+3]:.2f}"
            + "$\pm$"
            + f"{stds_all[i*4+3]:.2f}"
            + "\\\\ \n"
        )
    f.write("\\bottomline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
