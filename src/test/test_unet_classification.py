import os
import sys

sys.path.append("src/")

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.unet import Unet
from utils.data_utils import IFFT

# preprocess = torchvision.transforms.Resize((320, 320), antialias=True)
# postprocess = torchvision.transforms.Resize((640, 640), antialias=True)


def extract_patches(images, points, patch_size=32):
    """
    Extract patches from a point cloud
    :param images: torch.Tensor (B, C, H, W)
    :param points: torch.Tensor (B, N, 3)
    :param patch_size: int
    :return: torch.Tensor (B*patches, C, patch_size, patch_size)
    """

    size = (1, 1, patch_size, patch_size)
    grid = (
        F.affine_grid(
            torch.eye(2, 3).unsqueeze(0) * 0.2, size=size, align_corners=False
        ).view(1, 1, -1, 2)
        + points.unsqueeze(0).unsqueeze(2)
    ).to(images.device)

    patches = F.grid_sample(images.float(), grid.float(), align_corners=True).view(
        -1, images.shape[1], patch_size, patch_size
    )

    return patches


def pixel_coords_to_normalized(coords, im_size):
    """
    Convert coordinates from image format to bounding box format
    """
    x, y = coords
    x = x / im_size[1]
    y = y / im_size[0]
    return x, y


def evaluate_patches(image1, image2, boxes):
    from utils.evaluate import calmetric2D

    if len(boxes) == 0:
        return calmetric2D(image1, image2)

    metrics = {}
    psnrs = []
    ssims = []
    rmses = []
    haarpsis = []

    patches_1 = extract_patches(image1[None, None], torch.stack(boxes), patch_size=124)
    patches_2 = extract_patches(image2[None, None], torch.stack(boxes), patch_size=124)

    for i in range(patches_1.shape[0]):

        psnr, ssim, haarpsi, rmse = calmetric2D(patches_1[i][None], patches_2[i][None])

        psnrs.append(psnr)
        ssims.append(ssim)
        rmses.append(rmse)
        haarpsis.append(haarpsi)

    metrics["ssim"] = torch.mean(torch.tensor(ssims))
    metrics["psnr"] = torch.mean(torch.tensor(psnrs))
    metrics["haarpsi"] = torch.mean(torch.tensor(haarpsis))
    metrics["rmse"] = torch.mean(torch.tensor(rmses))

    return metrics


def coords_to_pixel(coords, im_size):
    """
    Convert coordinates from bounding box format to image format
    """
    x, y, w, h = coords
    x = int(x * im_size[1])
    y = int(y * im_size[0])
    w = int(w * im_size[1])
    h = int(h * im_size[0])
    return x, y, w, h


def motion_test_detection_data(path, annotations_path):

    # load model

    unet = Unet(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=6,
        batchnorm=nn.InstanceNorm2d,
        drop_prob=0.0,
    ).cuda()
    unet.load_state_dict(
        torch.load("src/model_weights/unet_denoising_detection_task.pth")
    )
    unet.eval()

    bboxes = []
    size = 640

    file_names = np.sort(os.listdir(path))

    for annotation_file in os.listdir(annotations_path):
        with open(os.path.join(annotations_path, annotation_file), "r") as f:
            lines = f.readlines()
            boxes_gt = []
            for line in lines:
                line = line.strip().split()
                # Format: ['', class, center_x, center_y, width, height],  remove the first element
                # class_id, center_x, center_y, width, height = map(float, line[1:])
                class_id, center_x, center_y, width, height = map(float, line)
                center_x = center_x * 2 - 1
                center_y = center_y * 2 - 1

                # center_x, center_y, width, height = box_convert(torch.tensor(coords_to_pixel([center_x, center_y, width, height], (size, size))), 'cxcywh', 'xyxy')
                boxes_gt.append(torch.tensor([center_x, center_y]))
            # check if tensorlist is empty
            if len(boxes_gt) > 0:
                bboxes.append(boxes_gt)
            f.close()

    if os.path.exists("results") == False:
        os.mkdir("results")

    scenarioius = ["light", "heavy"]
    metrics_all = defaultdict(list)

    for scenario in scenarioius:
        print("Scenario: ", scenario)
        data_path = (
            "Dataset/DetectionData/test/images_"
            + scenario
            + "/_detection_test_data.pth"
        )

        output_path = data_path.split("/_detection")[0] + "_corrected_unet"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        data_test = torch.load(data_path)

        kspaces_test = data_test["kspace_motion"]
        images_gt = data_test["image_rss"]

        metrics = []
        batch_test, H, W = kspaces_test.shape

        bsz = 1
        channels = 1

        motion_image = []
        moco_image = []
        gt_image = []
        metrics = []
        for idx_test, file_name in zip(range(batch_test), file_names):
            if not file_name.endswith(".png"):
                continue
            print("Processing: ", file_name)
            # try:
            with torch.no_grad():
                k_space_test = kspaces_test[idx_test].view(bsz, channels, H, W).cuda()
                img_motion_test = IFFT(k_space_test)

                image_gt = images_gt[idx_test].abs().cuda()
                gt_image.append(image_gt.detach().cpu())

                scale = IFFT(k_space_test).abs().std()
                img_motion_test = img_motion_test.abs().div(scale)

                unet_image = (unet(img_motion_test)).squeeze()

            moco_image.append(unet_image.detach().cpu())
            motion_image.append(img_motion_test.abs().detach().cpu())
            patch_metrics = evaluate_patches(
                unet_image.detach().abs(), image_gt.squeeze(), bboxes[idx_test]
            )

            # make a dictionary with the metrics
            metrics.append(
                {
                    "ssim": patch_metrics["ssim"],
                    "psnr": patch_metrics["psnr"],
                    "haar_psi": patch_metrics["haarpsi"],
                    "rmse": patch_metrics["rmse"],
                }
            )
            # normalize the image
            motion_corr = unet_image.detach().squeeze().cpu()
            motion_corr = (
                (motion_corr - motion_corr.min())
                / (motion_corr.max() - motion_corr.min())
                * 255.0
            )
            torchvision.io.write_png(
                motion_corr.to(torch.uint8).unsqueeze(0),
                os.path.join(output_path, file_name),
            )
            # except:
            #    print('Error in file: ', file_name)
        data = {
            "gt_image": torch.stack(gt_image),
            "moco_image": torch.stack(moco_image),
            "motion_image": torch.stack(motion_image),
            "metrics": metrics,
        }

        # save the dict as torch file
        torch.save(data, output_path + "/unet_denoising_detection_task.pth")
        # append the metrics to the metrics_all list
        metrics_all[scenario] = metrics

    return data, metrics_all


scenarios = ["light", "heavy"]
data, metrics_all = motion_test_detection_data(
    "Dataset/DetectionData/test/images", "Dataset/DetectionData/test/labels"
)
