# IM-MoCo

Official PyTorch implementation of IM-MoCo: Self-supervised MRI Motion Correction using Motion-Guided Implicit Neural Representations, submitted at MICCAI 2024

[Paper](files/MICCAI24_20240307_final.pdf) | [Supplementary Material](files/MICCAI24_20240307_Supplementary.pdf) | [Project Page]()

## Introduction

In this work, we propose a self-supervised MRI motion correction method that leverages motion-guided implicit neural representations to learn the motion patterns and correct the motion artifacts in MRI scans. The pre-trained klD-Net takes a motion-corrupted k-space
and outputs a motion mask, which is post-processed to yield a list of movement groups
indicated by the different colors in $S$ for better visualization. The number of movements
and corresponding lines guide the Motion INR and the Image INR. The Image INR
predicts the motion-free image and the Motion INR is used for guidance by optimizing
the forward motion yielding a motion-corrupted k-space. The discrepancy between the
motion-corrupted and the measured k-space is minimized using the data-consistency
(DC) loss. The gradient entropy is a denoiser on the Image-INR output to impose crisp
image priors. The final motion-corrected image is the output of the Image INR.

![ ](files/IM-MoCo_arch.png)

## Example Results

We assess the Image quality of the motion correction compared to state-of-the-art methods for two motion scenarios.

![ ](files/motion_correction_comp.png)

## Installation

### Requirements

- Python
- PyTorch
- torchvision
- h5py
- numpy
- tqdm
- scikit-learn
- scipy

### Install via conda/mamba

```bash
 mamba env create -f environment.yaml
```

For Hash-grid encoding we need to install [tiny-cuda](https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch) in the activated environment as well:

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Usage

### Dataset

- Download the NYU [fastMRI](https://fastmri.med.nyu.edu) $T_2$\_weighted Brain dataset.
- Run the following scripts to prepare the datasets(Todo: 🚧 Under construction 🚧):

```bash
python src/utils/prepareData.py
```

## Train

For training please refer to the `src/train` folder and for example run:

```bash
python src/train/train_kld_net.py
```

## Test

To run tests please run the scripts under `src/test` as follows:

```bash
python src/test/test_immoco.py
```

## Citation (Placeholder)

If you find this work helpful for your research, please cite the following paper:

```
@inproceedings{IM-MoCo,
  title={IM-MoCo: Self-supervised MRI Motion Correction using Motion-Guided Implicit Neural Representations},
  author={Author 1, Author 2, Author 3},
  booktitle={MICCAI},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
