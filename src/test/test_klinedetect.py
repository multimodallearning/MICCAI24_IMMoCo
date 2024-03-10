import os
import sys

sys.path.append("src/")

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from models.KLineDetect import get_unet
from utils.evaluate import dice_coef, iou_coef, metrics_classification

# load model
net = get_unet(
    in_chans=2, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0
).cuda()
net.load_state_dict(torch.load("src/model_weghts/kLDNet.pth"))

if os.path.exists("results/kLDNet") == False:
    os.mkdir("results/kLDNet")

scenarioius = ["light", "heavy"]
metrics_all = []


for scenario in scenarioius:
    print("Loading data...")
    data_path = "src/../Dataset/Brain/t2/test_files/_test_data_" + scenario + ".pth"
    data_test = torch.load(data_path)

    kspaces_test = data_test["kspace_motion"]

    metrics = []
    batch_test, H, W = kspaces_test.shape

    bsz = 1
    channels = 1
    net.eval()

    for idx_test in trange(batch_test):

        with torch.no_grad():

            k_space_test = kspaces_test[idx_test].view(bsz, channels, H, W).cuda()

            mask_test = data_test["mask"][idx_test].view(bsz, 1, H, W).cuda()

            k_space_test = (
                torch.view_as_real(k_space_test.view(bsz, 1, H, W))
                .squeeze(1)
                .permute(0, 3, 1, 2)
            )

            preds_test = net(k_space_test)

            output_test = preds_test.squeeze()

            dice_score = dice_coef(
                output_test.sigmoid() > 0.5, mask_test.squeeze().float()
            )
            iou = iou_coef(output_test.sigmoid() > 0.5, mask_test.squeeze().float())
            calassification_metrics = metrics_classification(
                (output_test.sigmoid() > 0.5).float(), mask_test.squeeze().float()
            )

            metrics.append(
                {
                    "dice": dice_score,
                    "iou": iou,
                    "sensitivity": calassification_metrics["Sensistivity"],
                    "specificity": calassification_metrics["Specificity"],
                    "presission": calassification_metrics["Presission"],
                    "f1": calassification_metrics["F1"],
                }
            )

    metrics_all.append(metrics)

# save metrics
torch.save(metrics_all, "results/kLDNet/kLDNet_metrics.pth")

# load the metrics and loop over the list of lists of dictionaries, get the means and stds of each scenario for each metric, plot the violin plots and save them as a latex table
metrics_all = torch.load("results/kLDNet/kLDNet_metrics.pth")

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

metrics_for_plot = torch.stack(metrics_for_plot).squeeze().view(12, 50)
means_all = torch.stack(means_all)
stds_all = torch.stack(stds_all)

# save means +- stds in a latex table
print("Saving latex table...")
with open("results/kLDNet/KLineDetect_metrics.tex", "w") as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{KLineDetect metrics}\n")
    f.write("\\label{tab:KLineDetect_metrics}\n")
    f.write("\\begin{tabular}{l|cccccc}\n")
    f.write("\\topline\n")
    f.write(
        "Scenario & Dice & IoU & Sensitivity & Specificity & Precision & F1 \\\\ \n"
    )
    f.write("\\midline\n")
    for i in range(len(scenarioius)):
        f.write(
            scenarioius[i]
            + " & "
            + f"{(means_all[i*6].item()):.3f} + {stds_all[i*6].item():.3f}"
            + " & "
            + f"{(means_all[i*6+1].item()):.3f} + {stds_all[i*6+1].item():.3f}"
            + " & "
            + f"{(means_all[i*6+2].item()):.3f} + {stds_all[i*6+2].item():.3f}"
            + " & "
            + f"{(means_all[i*6+3].item()):.3f} + {stds_all[i*6+3].item():.3f}"
            + " & "
            + f"{(means_all[i*6+4].item()):.3f} + {stds_all[i*6+4].item():.3f}"
            + " & "
            + f"{(means_all[i*6+5].item()):.3f} + {stds_all[i*6+5].item():.3f}"
            + "\\\\ \n"
        )
        f.write("\\bottomline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

# plot violin plots
# # plot all metrics for metrics all moderate in one violin plot for each metric
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 22, "font.family": "serif"})

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].violinplot(
    [metrics_for_plot[0]], positions=[1], showmeans=True, showmedians=True
)
axs[0].violinplot(
    [metrics_for_plot[1]], positions=[2], showmeans=True, showmedians=True
)
axs[0].set_title("Dice", fontsize=20)
axs[0].set_xticks([1, 2])
axs[0].set_xticklabels(["light", "heavy"], fontsize=20)
axs[0].set_ylabel("Score", fontsize=20)
axs[0].tick_params(labelsize=20)
axs[0].set_yticks([0.95, 0.98, 1.0])

axs[1].violinplot(
    [metrics_for_plot[2]], positions=[1], showmeans=True, showmedians=True
)
axs[1].violinplot(
    [metrics_for_plot[3]], positions=[2], showmeans=True, showmedians=True
)
axs[1].set_title("IoU", fontsize=20)
axs[1].set_xticks([1, 2])
axs[1].set_xticklabels(["Light", "Heavy"], fontsize=20)
axs[1].set_yticks([0.95, 0.98, 1.0])
axs[1].tick_params(
    labelsize=20, top=False, bottom=True, left=False, right=False, labelleft=False
)

axs[2].violinplot(
    [metrics_for_plot[4]], positions=[1], showmeans=True, showmedians=True
)
axs[2].violinplot(
    [metrics_for_plot[5]], positions=[2], showmeans=True, showmedians=True
)
axs[2].set_title("F1-Score", fontsize=20)
axs[2].set_xticks([1, 2])
axs[2].set_xticklabels(["Light", "Heavy"], fontsize=20)
axs[2].set_yticks([0.95, 0.98, 1.0])
axs[2].tick_params(
    labelsize=20, top=False, bottom=True, left=False, right=False, labelleft=False
)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.3)
# save figure
fig.savefig("results/kLDNet/KLineDetect_metrics.png", bbox_inches="tight", dpi=300)
plt.close(fig)
