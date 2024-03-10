# IM-MoCo

Official PyTorch implementation of IM-MoCo: Self-supervised MRI Motion Correction using Motion-Guided Implicit Neural Representations, submitted at MICCAI 2024

[Paper](files/MICCAI24_20240307_final.pdf) | [Supplementary Material](files/MICCAI24_20240307_Supplementary.pdf) | [Project Page]()

## Introduction

This repository contains the official PyTorch implementation of IM-MoCo: Self-supervised MRI Motion Correction using Motion-Guided Implicit Neural Representations. In this work, we propose a self-supervised MRI motion correction method that leverages motion-guided implicit neural representations to learn the motion patterns and correct the motion artifacts in MRI scans. We propose a contrastive learning framework that learns representations of motion artifacts and corrects them using a motion-guided implicit neural representation. Our method outperforms the state-of-the-art methods for MRI motion correction in terms of reconstruction quality and motion correction.

![ ](files/IM-MoCo_arch.png)

## Example Results

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

Please prepare your dataset in the following format:

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
