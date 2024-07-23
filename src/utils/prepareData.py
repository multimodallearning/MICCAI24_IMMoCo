# %%
import sys

sys.path.append("src/")
import os
import re
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm

from utils.data_utils import *
from utils.evaluate import *
from utils.motion_utils import *


def getFieldStrength(h_file):
    ismhd = h_file["ismrmrd_header"][()]
    match = re.findall(
        r"<systemFieldStrength_T>\d\.\d*</systemFieldStrength_T>", str(ismhd)
    )[0]
    match = match.replace("<systemFieldStrength_T>", "")
    match = match.replace("</systemFieldStrength_T>", "")
    return 1.5 if float(match) < 2.0 else 3.0


def getAcquisitionType(h_file):
    return "PD" if h_file.attrs["acquisition"] == "CORPD_FBK" else "PDFS"


def create_data_split(PATH="Dataset/FastMRI/t2"):

    print("Crating Train Set")
    Train_PATH = os.path.join(PATH, "train/")
    hf_map = {}
    for fname in tqdm(os.listdir(Train_PATH)):
        with h5py.File(os.path.join(Train_PATH, fname)) as hf:
            hf_map[fname] = (getFieldStrength(hf), getAcquisitionType(hf))

    PD_1T = [f for f, v in hf_map.items() if v == (1.5, "PD")]
    PD_3T = [f for f, v in hf_map.items() if v == (3.0, "PD")]
    PDFS_1T = [f for f, v in hf_map.items() if v == (1.5, "PDFS")]
    PDFS_3T = [f for f, v in hf_map.items() if v == (3.0, "PDFS")]

    train_list = np.concatenate([PDFS_1T, PDFS_3T])

    train_list = np.random.choice(train_list, 200, replace=False)

    train_path = "../Dataset/Brain/t2/train_files/"
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    with h5py.File(train_path + "_train_data.h5", "w") as f:
        for fname in tqdm(train_list):
            hf = h5py.File(os.path.join(Train_PATH, fname))
            f.create_dataset(fname, data=hf["kspace"][:])
            hf.close()

    # validation set
    print("Creating Validation set")
    VAL_PATH = os.path.join(PATH, "val/")
    hf_map = {}

    for fname in tqdm(os.listdir(VAL_PATH)):
        with h5py.File(os.path.join(VAL_PATH, fname)) as hf:
            hf_map[fname] = (getFieldStrength(hf), getAcquisitionType(hf))

    PD_1T = [f for f, v in hf_map.items() if v == (1.5, "PD")]
    PD_3T = [f for f, v in hf_map.items() if v == (3.0, "PD")]
    PDFS_1T = [f for f, v in hf_map.items() if v == (1.5, "PDFS")]
    PDFS_3T = [f for f, v in hf_map.items() if v == (3.0, "PDFS")]

    val_list = np.concatenate([PDFS_1T, PDFS_3T])

    val_list = np.random.choice(val_list, 50, replace=False)

    val_path = "../Dataset/Brain/t2/val_files/"
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    with h5py.File(val_path + "_val_data.h5", "w") as f:
        for fname in tqdm(val_list):
            hf = h5py.File(os.path.join(VAL_PATH, fname))
            f.create_dataset(fname, data=hf["kspace"][:])
            hf.close()

    print("Crating Test Set")
    Test_PATH = os.path.join(PATH, "test/")
    hf_map = {}

    for fname in tqdm(os.listdir(Test_PATH)):
        with h5py.File(os.path.join(Test_PATH, fname)) as hf:
            hf_map[fname] = (getFieldStrength(hf), getAcquisitionType(hf))

    PD_1T = [f for f, v in hf_map.items() if v == (1.5, "PD")]
    PD_3T = [f for f, v in hf_map.items() if v == (3.0, "PD")]
    PDFS_1T = [f for f, v in hf_map.items() if v == (1.5, "PDFS")]
    PDFS_3T = [f for f, v in hf_map.items() if v == (3.0, "PDFS")]

    test_list = np.concatenate([PDFS_1T, PDFS_3T])

    test_list = np.random.choice(test_list, 51, replace=False)

    test_path = "../Dataset/Brain/t2/test_files/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    with h5py.File(test_path + "_test_data.h5", "w") as f:
        for fname in tqdm(test_list):
            hf = h5py.File(os.path.join(Test_PATH, fname))
            f.create_dataset(fname, data=hf["kspace"][:])
            hf.close()


def preprocess_dataset(path="Dataset/Brain/t2/train_files/_train_data.h5"):

    data_file = h5py.File(path)

    kspaces = []

    for file_name in tqdm(sorted(list(data_file.keys()))):
        ks = torch.from_numpy(data_file[file_name][()][1])

        kspace, _ = prepare_data(ks)

        # if the shapes are not 320x320, then skip the sample
        if kspace.shape != torch.Size([320, 320]):
            print(kspace.shape)
            print("Skipping sample with shape: ", kspace.shape)
            continue

        kspaces.append(kspace)

    # create a dictionary for all lists that are tansformed to tensors
    # make tensors from the the lists from the for loop
    kspaces = torch.stack(kspaces).squeeze()

    data = {"kspace": kspaces}

    # save the dict as torch file
    torch.save(data, path)


def motion_test_data(path):
    test_data_file = h5py.File(path)

    scenarios = ["light", "heavy"]
    movements = [np.arange(6, 10), np.arange(16, 20)]
    metrics_all = defaultdict(list)
    for scenario, movement in zip(scenarios, movements):
        scenario_path = path.split(".h5")[0] + "_" + scenario + ".pth"
        metrics = []
        kspaces_motion = []
        images_rss = []
        rotations = []
        translations = []
        masks = []
        for file_name in tqdm(sorted(list(test_data_file.keys()))):
            ks = torch.from_numpy(test_data_file[file_name][()][0])

            kspace, image_rss = prepare_data(ks)
            num_movements = np.random.choice(movement, 1, replace=True)[0]
            # if the shapes are not 320x320, then skip the sample
            if kspace.shape != torch.Size([320, 320]):
                print(kspace.shape)
                print("Skipping sample with shape: ", kspace.shape)
                continue

            kspace_motion, mask, rotation, translation = motion_simulation2D(
                IFFT(kspace), num_movements
            )

            kspaces_motion.append(kspace_motion)
            images_rss.append(image_rss)
            rotations.append(rotation)
            translations.append(translation)
            masks.append(mask)

            H, W = kspace_motion.shape

            crop = [int(H / 4), int(W / 4)]
            image_gt_crop = image_rss.abs()[crop[0] : -crop[0], crop[1] : -crop[1]]
            motion_corrupted_crop = IFFT(kspace_motion).abs()[
                crop[0] : -crop[0], crop[1] : -crop[1]
            ]

            psnr, ssim, haar_psi, rmse = calmetric2D(
                motion_corrupted_crop[None, None], image_gt_crop[None, None]
            )

            # make a dictionary with the metrics
            metrics.append(
                {"ssim": ssim, "psnr": psnr, "haar_psi": haar_psi, "rmse": rmse}
            )

        # create a dictionary for all lists that are tansformed to tensors
        # make tensors from the the lists from the for loop
        kspace_motion = torch.stack(kspaces_motion).squeeze()
        image_rss = torch.stack(images_rss).squeeze()
        rotation = rotations
        translation = translations
        mask = torch.stack(masks).squeeze()

        data = {
            "kspace_motion": kspace_motion,
            "image_rss": image_rss,
            "rotation": rotation,
            "translation": translation,
            "mask": mask,
            "metrics": metrics,
        }

        # save the dict as torch file
        torch.save(data, scenario_path)
        # append the metrics to the metrics_all list
        metrics_all[scenario] = metrics


def main():

    DATA_PATH = "Dataset/FastMRI/t2"

    TRAIN_PATH = "Dataset/Brain/t2/train_files/_train_data.h5"
    VAL_PATH = "Dataset/Brain/t2/train_files/_train_data.h5"
    TEST_PATH = "Dataset/Brain/t2/train_files/_train_data.h5"

    create_data_split(DATA_PATH)

    preprocess_dataset(TRAIN_PATH)
    preprocess_dataset(VAL_PATH)

    motion_test_data(TEST_PATH)

    print()

    scenarios = ["light", "heavy"]

    for scenario in scenarios:
        path_metrics = TEST_PATH + "/_test_data_" + scenario + ".pth"
        metrics = torch.load(path_metrics)["metrics"]

        print("Scenario: ", scenario)
        val_psnr = [d["psnr"] for d in metrics]
        val_ssim = [[d["ssim"] for d in metrics]]
        val_haar = [d["haar_psi"] for d in metrics]
        val_rmse = [[d["rmse"] for d in metrics]]

        val_psnr = np.array(val_psnr)
        val_ssim = np.array(val_ssim)
        val_haar = np.array(val_haar)
        val_rmse = np.array(val_rmse)

        print(f"PSNR: {val_psnr.mean():.2f} \pm {val_psnr.std():.2f}")
        print(f"SSIM: {val_ssim.mean()*100:.2f} \pm {val_ssim.std()*100:.2f}")
        print(f"RMSE: {val_rmse.mean()*100:.2f} \pm {val_rmse.std()*100:.2f}")
        print(f"Haar: {val_haar.mean()*100:.2f} \pm {val_haar.std()*100:.2f}")
