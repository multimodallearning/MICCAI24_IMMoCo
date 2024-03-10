import sys

sys.path.append("src/")

import torch
import torch.nn.functional as F
import torchvision

from models.classification import get_model
from utils.classification_utils import multi_acc


def test_classification(
    PATH="../../Dataset/DetectionData/test/images/dataset_test.pth",
):
    # test the model
    test_dataset = torch.load(PATH)
    test_img = test_dataset["images"]
    test_label = test_dataset["labels"]

    print("Test labels:", test_label.shape)

    # # normalize images between 0 and 1
    min_test = (
        test_img.view(test_img.shape[0], -1)
        .min(1)
        .values.unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
    )
    max_test = (
        test_img.view(test_img.shape[0], -1)
        .max(1)
        .values.unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
    )
    test_img = (test_img - min_test) / (max_test - min_test)

    # ResNet Specific normalization: https://pytorch.org/hub/pytorch_vision_resnet/
    process = torchvision.transforms.Compose(
        [
            torchvision.transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    test_img = process(test_img)

    num_classes = test_label.unique().shape[0]

    model = get_model(num_classes).cuda()
    model.load_state_dict(torch.load("src/model_weights/classification_model.pth"))
    model.eval()

    test_label = F.one_hot(test_label, num_classes=test_label.unique().shape[0]).float()
    with torch.no_grad():
        outputs = model(test_img.cuda())
        acc = multi_acc(outputs, test_label.cuda())

    return acc


def main():

    motion_free_path = "../../Dataset/DetectionData/test/images/dataset_test.pth"
    light_motion_path = "../../Dataset/DetectionData/test/images_light/dataset_test.pth"
    heavy_motion_path = "../../Dataset/DetectionData/test/images_light/dataset_test.pth"

    light_immoco_path = (
        "../../Dataset/DetectionData/test/images_light_immoco/dataset_test.pth"
    )
    heavy_immoco_path = (
        "../../Dataset/DetectionData/test/images_heavy_immoco/dataset_test.pth"
    )

    light_unet_path = (
        "../../Dataset/DetectionData/test/images_light_unet/dataset_test.pth"
    )
    heavy_unet_path = (
        "../../Dataset/DetectionData/test/images_heavy_unet/dataset_test.pth"
    )

    acc_mot_free = test_classification(motion_free_path)
    print("Test Accuracy for Motion-free: %.2f %%" % acc_mot_free.item())
    acc_light_mot = test_classification(light_motion_path)
    print("Test Accuracy for Light motion: %.2f %%" % acc_light_mot.item())
    acc_heav_mot = test_classification(heavy_motion_path)
    print("Test Accuracy for Heavy motion: %.2f %%" % acc_heav_mot.item())

    acc_immoco_light = test_classification(light_immoco_path)
    print("Test Accuracy for IM-MoCo light: %.2f %%" % acc_immoco_light.item())
    acc_immoco_heavy = test_classification(heavy_immoco_path)
    print("Test Accuracy for IM-MoCo heavy: %.2f %%" % acc_immoco_heavy.item())

    acc_unet_light = test_classification(light_unet_path)
    print("Test Accuracy for U-Net light: %.2f %%" % acc_unet_light.item())
    acc_unet_heacy = test_classification(heavy_unet_path)
    print("Test Accuracy for U-Net heavy: %.2f %%" % acc_unet_heacy.item())


if __name__ == "__main__":
    main()
