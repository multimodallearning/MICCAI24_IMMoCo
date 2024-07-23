"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import csv
import os

import __main__
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from piq import haarpsi, ssim


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize array to [0, 1] range"""
    if x.shape[0] > 1:
        # batchwise normalization
        max_b = x.view(x.shape[0], -1).max(1).values
        min_b = x.view(x.shape[0], -1).min(1).values
        return (x - min_b.view(-1, 1, 1, 1)) / (
            (max_b - min_b).view(-1, 1, 1, 1) + 1e-24
        )
    else:
        return (x - x.min()) / (x.max() - x.min() + 1e-24)


def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute RMSE between two arrays"""
    return torch.sqrt(torch.mean((x - y) ** 2))


def my_psnr(img1, img2, data_range=None, reduction="mean"):
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    if data_range is None:
        max_pixel = img2.view(img2.shape[0], -1).max(1).values
    else:
        max_pixel = data_range

    if reduction == "none":
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))
    else:
        return (20 * torch.log10(max_pixel / torch.sqrt(mse))).mean()


# def sharpnes_metric(x: torch.Tensor) -> torch.Tensor:
#     """Compute sharpness metric between two arrays"""
#     import cpbd

#     return cpbd.compute(x.squeeze().numpy() * 255.0, mode="sharpness")


def calmetric2D(pred_recon, gt_recon):
    # check sizes -> (B, C, H, W )
    if not pred_recon.ndim == 4 or not gt_recon.ndim == 4:
        raise ValueError("Input tensors must be 4D")

    # normalize
    pred = normalize(pred_recon)
    gt = normalize(gt_recon)

    ssim_kernel = 11
    haar_scale = 3
    if pred.shape[-1] < ssim_kernel or pred.shape[-2] < ssim_kernel:
        ssim_kernel = min(pred.shape[-1], pred.shape[-2], ssim_kernel) - 1
        haar_kernel = min(pred.shape[-1], pred.shape[-2], haar_kernel) - 1
        haar_scale = int(np.log2(haar_kernel))

    psnr_array = my_psnr(pred, gt, data_range=1.0, reduction="mean")
    ssim_array = ssim(
        pred, gt, data_range=1.0, kernel_size=ssim_kernel, reduction="mean"
    )
    haar_psi_array = haarpsi(pred, gt, scales=haar_scale, reduction="mean")
    rmse_array = rmse(pred, gt)

    return psnr_array, ssim_array, haar_psi_array, rmse_array


def calmetric3D(pred_recon, gt_recon):

    batch = pred_recon.shape[0]

    ssim_array = np.zeros(batch)
    psnr_array = np.zeros(batch)
    haar_psi = np.zeros(batch)
    rmse_array = np.zeros(batch)

    for i in range(batch):
        psnr_array[i], ssim_array[i], haar_psi[i], rmse_array[i] = calmetric2D(
            pred_recon[i].unsqueeze(0), gt_recon[i].unsqueeze(0)
        )

    return psnr_array.mean(), ssim_array.mean(), haar_psi.mean(), rmse_array.mean()


def calc_metrics(pred_recon, gt_recon, save_path, name):
    # check sizes -> (B, C, D, H, W )
    if not pred_recon.ndim == 5 or not gt_recon.ndim == 5:
        batch = pred_recon.shape[0]
        if batch == 1:
            psnr_array, ssim_array, haar_psi, rmse_array = calmetric2D(
                pred_recon, gt_recon
            )
            print(
                f"PSNR: {psnr_array} db, SSIM: {ssim_array*100}%, HaarPSI: {haar_psi}, RMSE: {rmse_array}"
            )
            with open(os.path.join(save_path, name + ".csv"), "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["PSNR", "SSIM", "HaarPSI", "RMSE"])
                writer.writerow([psnr_array, ssim_array, haar_psi, rmse_array])
        else:
            psnr_array, ssim_array, haar_psi, rmse_array = calmetric3D(
                pred_recon, gt_recon
            )
            print(
                f"PSNR: {psnr_array} db, SSIM: {ssim_array*100}%, HaarPSI: {haar_psi}, RMSE: {rmse_array}"
            )
            with open(os.path.join(save_path, name + ".csv"), "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["PSNR", "SSIM", "HaarPSI", "RMSE"])
                writer.writerow([psnr_array, ssim_array, haar_psi, rmse_array])
        # save as latex table
        df = pd.read_csv(os.path.join(save_path, name + ".csv"))
        df = df.round(3)
        df.to_latex(os.path.join(save_path, name + ".tex"), index=False, escape=False)
        os.remove(os.path.join(save_path, name + ".csv"))
        print("Saved metrics as csv and latex table in results folder.")

    elif pred_recon.ndim == 5 or gt_recon.ndim == 5:
        batch = pred_recon.shape[0]
        psnr_array = np.zeros(batch)
        ssim_array = np.zeros(batch)
        haar_psi = np.zeros(batch)
        rmse_array = np.zeros(batch)
        for i in range(batch):
            psnr_array[i], ssim_array[i], haar_psi[i], rmse_array[i] = calmetric2D(
                pred_recon[i].unsqueeze(0), gt_recon[i].unsqueeze(0)
            )
        print(
            f"PSNR: {psnr_array.mean()} db, SSIM: {ssim_array.mean()*100}%, HaarPSI: {haar_psi.mean()}, RMSE: {rmse_array.mean()}"
        )
        with open(os.path.join(save_path, name + ".csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["PSNR", "SSIM", "HaarPSI", "RMSE"])
            writer.writerow(
                [
                    psnr_array.mean(),
                    ssim_array.mean(),
                    haar_psi.mean(),
                    rmse_array.mean(),
                ]
            )
        # save as latex table
        df = pd.read_csv(os.path.join(save_path, name + ".csv"))
        df = df.round(3)
        df.to_latex(os.path.join(save_path, name + ".tex"), index=False, escape=False)
        os.remove(os.path.join(save_path, name + ".csv"))
        print("Saved metrics as csv and latex table in results folder.")


# write a function that creates a violin plot for the metrics of the whole dataset
def create_violin_plot(
    data, method_names, metric_name="SSIM", save_path="./", name="violin_plot"
):
    # chech if data is numpy array and length of method_names is correct
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if not len(method_names) == data.shape[1]:
        raise ValueError(
            "length of method_names must be equal to the number of methods in data"
        )

    # get the metrics as numpy array and create a dataframe
    df = pd.DataFrame(data, columns=method_names)
    df = df.round(3)
    sns.set_style("darkgrid")
    # create a color pallete suited for scientific plots
    my_palette = sns.color_palette("colorblind", 4)

    sns.set_palette(my_palette)
    violin_plot = sns.violinplot(data=df)
    violin_plot.set_title(metric_name)
    violin_plot.set_ylabel(metric_name)
    violin_plot.set_xlabel("Methods")
    violin_plot.get_figure().savefig(os.path.join(save_path, name + ".png"))

    print("Saved violin plot in results folder.")


def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred))
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = torch.mean((2.0 * intersection + smooth) / (union + smooth), dim=0)
    return dice


def sensitivity_score(true_positives: torch.Tensor, y_gt: torch.Tensor):
    sensitivity = true_positives / (y_gt == 1).sum()
    return sensitivity


def presission_score(true_positives: torch.Tensor, false_positives: torch.Tensor):
    presission = true_positives / (true_positives + false_positives)
    return presission


def specificity_score(true_negatives: torch.Tensor, y_gt: torch.Tensor):
    specificity = true_negatives / (y_gt == 0).sum()
    return specificity


def f1_score(presission: torch.Tensor, sensitivity: torch.Tensor):
    f1 = 2 * (presission * sensitivity) / (presission + sensitivity)
    return f1


def metrics_classification(y_pred: torch.Tensor, y_gt: torch.Tensor):
    # calculate scores for statistics
    true_positives = torch.sum(torch.logical_and((y_pred == 1), (y_gt == 1)))
    false_positives = torch.sum(torch.logical_and((y_pred == 1), (y_gt == 0)))
    true_negative = torch.sum(torch.logical_and((y_pred == 0), (y_gt == 0)))
    metrics_dict = {}
    metrics_dict["Sensistivity"] = sensitivity_score(true_positives, y_gt).item()
    metrics_dict["Specificity"] = specificity_score(true_negative, y_gt).item()
    metrics_dict["Presission"] = presission_score(
        true_positives, false_positives
    ).item()
    metrics_dict["F1"] = f1_score(
        presission_score(true_positives, false_positives),
        sensitivity_score(true_positives, y_gt),
    ).item()

    return metrics_dict


if __name__ == "__main__":

    data = np.random.normal(size=(100, 2), loc=0, scale=100)

    method_names = ["method1", "method2"]

    create_violin_plot(data, method_names, save_path="./", name="test")
