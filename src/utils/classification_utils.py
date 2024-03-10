import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image


def filter_annotations(PATH="Annotations/brain.csv"):
    annotations = pd.read_csv(PATH)

    number_of_labels = annotations["label"].value_counts()

    # please refer to fastmri+ paper for label counts: https://arxiv.org/ftp/arxiv/papers/2109/2109.03812.pdf
    # filter annotations where the label has more than 1000 instances
    annotations_n = annotations[
        annotations["label"].isin(number_of_labels[number_of_labels > 1000].index)
    ]

    # since Posttreatment change is over 1000 but is ambagious for classification we filter it out
    annotations_n = annotations_n[annotations_n["label"] != "Posttreatment change"]

    # save it to new csv file
    annotations_n.to_csv("Annotations/brain_filtered.csv", index=False)


def map_brain_label_to_id(label_txt):
    brain_labels_dict = {
        "Nonspecific white matter lesion": 0,
        "Craniotomy": 1,
    }
    if label_txt not in brain_labels_dict:
        print(f"Warning: Unknown label '{label_txt}' encountered.")
        return -1
    return brain_labels_dict[label_txt]


def convert_annotations_to_yolo(df, fastmri_file, slice_number, output_path, img_shape):
    labels_for_slice = df[(df["file"] == fastmri_file) & (df["slice"] == slice_number)]
    # Write to .txt file
    annotation_filename = os.path.join(
        output_path, f"{fastmri_file}_slice_{slice_number}.txt"
    )
    # if the file alread exists, delete it
    if os.path.exists(annotation_filename):
        os.remove(annotation_filename)

    for _, row in labels_for_slice.iterrows():
        x0, y0, w, h = row["x"], row["y"], row["width"], row["height"]
        label_txt = row["label"]

        # Normalize the coordinates
        x_center = x0 + (w / 2)
        y_center = y0 + (h / 2)
        x_center /= img_shape[0]  # Normalize by image width
        y_center /= img_shape[1]  # Normalize by image height
        w /= img_shape[0]
        h /= img_shape[1]

        class_id = map_brain_label_to_id(label_txt)

        with open(annotation_filename, "a") as file:
            file.write(f"{class_id} {x_center} {y_center} {w} {h}\n")


def convert_Alldataset_to_yolo2(
    annotation_path, fastmri_path, output_image_path, output_label_path
):
    df = pd.read_csv(annotation_path, index_col=None, header=0)
    # get all filenames from df
    train_files = df["file"].unique().tolist()

    total_files = len(train_files)
    total_slices = 0
    slices_with_annotations = 0
    slices_without_annotations = 0
    errors = 0

    # Create the output directories if they don't exist
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)

    for i, fastmri_file in enumerate(train_files):
        # if the annotation x, y are empty then skip the file

        try:
            datafile = os.path.join(fastmri_path, fastmri_file + ".h5")
            with h5py.File(datafile, "r") as f:
                img_data = f["reconstruction_rss"][:]
                img_data = img_data[:, ::-1, :]  # flipped up down

            for slice_number, slice in enumerate(img_data):
                # if the slice exists in the annotation file
                if (df["file"] == fastmri_file).any() & (
                    df["slice"] == slice_number
                ).any():

                    total_slices += 1
                    labels_for_slice = df[
                        (df["file"] == fastmri_file) & (df["slice"] == slice_number)
                    ]

                    arrimg = np.squeeze(slice)
                    image_2d_scaled = (np.maximum(arrimg, 0) / arrimg.max()) * 255.0
                    image = Image.fromarray(np.uint8(image_2d_scaled))

                    # check if labels_for_slice is empty
                    if (
                        labels_for_slice is not None
                        and isinstance(labels_for_slice, pd.DataFrame)
                        and not labels_for_slice.empty
                    ):
                        slices_with_annotations += 1
                        image_filename = os.path.join(
                            output_image_path,
                            f"{fastmri_file}_slice_{slice_number}.png",
                        )
                        image.save(image_filename)
                        label_filename = os.path.join(
                            output_label_path,
                            f"{fastmri_file}_slice_{slice_number}.txt",
                        )
                        # Convert annotations to YOLO format and save label file
                        convert_annotations_to_yolo(
                            labels_for_slice,
                            fastmri_file,
                            slice_number,
                            output_label_path,
                            image.size,
                        )

                else:
                    slices_without_annotations += 1
                    print(f"slice {slice_number} does not exist in the annotation file")

            if (i + 1) % 10 == 0 or i + 1 == total_files:
                print(f"Processed {i + 1}/{total_files} files.")
        except Exception as e:
            print(f"Error processing file {fastmri_file}: {e}")
            errors += 1

    print(f"Total files processed: {total_files}")
    print(f"Total slices processed: {total_slices}")
    print(f"Slices with annotations: {slices_with_annotations}")
    print(f"Slices without annotations: {slices_without_annotations}")
    if errors > 0:
        print(f"Encountered errors in {errors} files.")
        print(f"Encountered errors in {errors} files.")


def create_and_split_data():

    annotation_path_train = "Annotations/brain_filtered.csv"
    fastmri_path_train = "Dataset/RowData/"
    output_image_path_train = "Dataset/ClassificationData/train/images/"
    output_label_path = "Dataset/ClassificationData/train/labels/"

    convert_Alldataset_to_yolo2(
        annotation_path_train,
        fastmri_path_train,
        output_image_path_train,
        output_label_path,
    )

    # get the train data dir and count file number: split into 80/20 for train and validation and move the drawn validation data into a new folder
    train_data_dir = "Dataset/ClassificationData/train/labels/"
    label_files = os.listdir(train_data_dir)
    images = "Dataset/ClassificationData/train/images/"
    num_data = len(label_files)

    # split the data into 80/20
    num_train = int(num_data * 0.8)
    num_val = num_data - num_train
    print(num_train, num_val)
    # get the list of files in the train data dir

    files = os.listdir(train_data_dir)
    files_img = [f.replace(".txt", ".png") for f in files]
    # if name contains slice0 in name

    os.makedirs("Dataset/ClassificationData/val/labels", exist_ok=True)
    os.makedirs("Dataset/ClassificationData/val/images", exist_ok=True)

    for i in range(num_val):
        os.rename(
            train_data_dir + files[i],
            "Dataset/ClassificationData/val/labels/" + files[i],
        )
        os.rename(
            images + files_img[i],
            "Dataset/ClassificationData/val/images/" + files_img[i],
        )

    val_data_dir = "Dataset/ClassificationData/val/labels/"
    label_files = os.listdir(val_data_dir)
    images = "Dataset/ClassificationData/val/images/"
    num_data = len(label_files)
    # take 50 random images that contain slice0 in name
    num_test = 50
    files = os.listdir(val_data_dir)
    files = [
        f
        for f in files
        if "slice_1.txt" in f or "slice_0.txt" in f or "slice_2.txt" in f
    ]

    files_img = [
        f.replace(".txt", ".png")
        for f in files
        if "slice_1.txt" in f
        or "slice_0.txt" in f
        or "slice_2.txt" in f
        or "slice_3.txt" in f
    ]

    print(len(files_img))
    os.makedirs("Dataset/ClassificationData/test/images", exist_ok=True)
    os.makedirs("Dataset/ClassificationData/test/labels", exist_ok=True)

    count = 0
    for i in range(num_test):
        os.rename(
            val_data_dir + files[i],
            "Dataset/ClassificationData/test/labels/" + files[i],
        )
        os.rename(
            images + files_img[i],
            "Dataset/ClassificationData/test/images/" + files_img[i],
        )
        count += 1


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


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

    patches_1 = extract_patches(image1[None, None], boxes, patch_size=124)
    patches_2 = extract_patches(image2[None, None], boxes, patch_size=124)

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


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    y_test = torch.argmax(y_test, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = acc * 100

    return acc


# read text file and iterate line by line
# Load ground truth annotations
def get_boxes(annotaions_path):
    ground_truth = defaultdict(list)
    for annotation_file in os.listdir(annotaions_path):
        with open(os.path.join(annotaions_path, annotation_file), "r") as f:
            lines = f.readlines()
            boxes_gt = []
            for line in lines:
                line = line.strip().split()
                # Format: ['', class, center_x, center_y, width, height]
                class_id, center_x, center_y, width, height = map(float, line)
                # normalize -1 and 1
                center_y = center_y * 2 - 1
                center_x = center_x * 2 - 1
                boxes_gt.append(torch.tensor([class_id, center_x, center_y]))

            # check if tensorlist is empty
            if len(boxes_gt) > 0 and len(boxes_gt) > 1:
                ground_truth[annotation_file[:-4]].append(torch.stack(boxes_gt, dim=0))
            elif len(boxes_gt) > 0 and len(boxes_gt) == 1:
                ground_truth[annotation_file[:-4]].append(boxes_gt[0].unsqueeze(0))

            f.close()
    return ground_truth


def process_dataset(image_path, annotaion_path, patch_size):
    """
    Process the dataset
    :param image_path: str
    :param annotation_path: str
    :param patch_size: int
    :return: torch.Tensor (B*patches, C, patch_size, patch_size)
    """
    # get root_path and create output path which is folder_name + '_patches'
    output_path = os.path.join(
        "ClassificationData",
        "dataset_" + image_path.split("/images")[0].split("/")[-1] + ".pth",
    )
    if not os.path.exists(output_path.split("/")[0]):
        os.makedirs(output_path.split("/")[0])

    print("Processing dataset and saving to", output_path)
    image_files = np.sort(os.listdir(image_path))
    annotations = get_boxes(annotaion_path)
    patches = []
    labels = []
    dataset = {}
    files_processed = 0
    for image_file in image_files:
        try:
            image = (
                torchvision.io.read_image(os.path.join(image_path, image_file))
                .unsqueeze(0)
                .float()
            )
            points = annotations[image_file[:-4]][0][:, 1:]
            label = annotations[image_file[:-4]][0][:, 0].long()
            patches.append(extract_patches(image, points, patch_size))
            labels.append(label)
            files_processed += 1
        except Exception as e:
            print(e)
            continue
    dataset["images"] = torch.cat(patches, dim=0)
    dataset["labels"] = torch.cat(labels, dim=0)
    print("Processed", files_processed, "files")

    # get label weights for training
    if "train" in image_path:
        unique, counts = torch.cat(labels, dim=0).unique(return_counts=True)
        label_weight = counts.min() / counts.float()
        dataset["label_weight"] = label_weight
    torch.save(dataset, output_path)


# train dataset
images_path = "Dataset/ClassificationData/train/images/"
annotations_path = "Dataset/Classification/train/labels/"
process_dataset(images_path, annotations_path, 124)

# # val dataset
images_path = "Dataset/CLassificationData/val/images/"
annotations_path = "../ClassificationData/val/labels/"
process_dataset(images_path, annotations_path, 124)

# test dataset
images_path = "Dataset/ClassificationData/test/images/"
annotations_path = "Dataset/ClassificationData/test/labels/"
process_dataset(images_path, annotations_path, 124)
