# IM-MoCo

Official PyTorch implementation of IM-MoCo: Self-supervised MRI Motion Correction using Motion-Guided Implicit Neural Representations, submitted at MICCAI 2024

[Paper](images/MICCAI24_20240307_final.pdf) | [Supplementary Material](images/MICCAI24_20240307_Supplementary.pdf) | [Project Page]()

## Introduction

This repository contains the official PyTorch implementation of IM-MoCo: Self-supervised MRI Motion Correction using Motion-Guided Implicit Neural Representations. In this work, we propose a self-supervised MRI motion correction method that leverages motion-guided implicit neural representations to learn the motion patterns and correct the motion artifacts in MRI scans. We propose a contrastive learning framework that learns representations of motion artifacts and corrects them using a motion-guided implicit neural representation. Our method outperforms the state-of-the-art methods for MRI motion correction in terms of reconstruction quality and motion correction.

![Images](images/IM-MoCo.png)

## Installation

### Requirements

- Python 3.6+ Al-Haj 
- PyTorch 1.7.0+
- torchvision 0.8.0+
- tensorboard
- h5py
- nibabel
- scikit-learn
- scipy
- numpy
- tqdm

### Install via pip

```bash
pip install -r requirements.txt
```

## Usage

### Dataset

Please prepare your dataset in the following format:

```
dataset
├── train
│   ├── image
│   │   ├── 1.h5
│   │   ├── 2.h5
│   │   └── ...
│   └── label
│       ├── 1.h5
│       ├── 2.h5
│       └── ...
└── val
    ├── image
    │   ├── 1.h5
    │   ├── 2.h5
    │   └── ...
    └── label
        ├── 1.h5
        ├── 2.h5
        └── ...
```

Each h5 file contains a 3D MRI scan in the following format:

```python
import h5py

with h5py.File('1.h5', 'r') as f:
    image = f['image'][:]
```

### Train

```bash
python train.py
```

### Test

```bash
python test.py
```

## Results

### Reconstruction Quality

We evaluate the reconstruction quality using the peak signal-to-noise ratio (PSNR) and the structural similarity index (SSIM).

| Method           | PSNR | SSIM |
| ---------------- | ---- | ---- |
| Motion-Corrupted | 20.3 | 0.78 |
| AF               | 25.6 | 0.88 |
| U-Nets           | 27.8 | 0.92 |
| AF+              | 28.5 | 0.94 |
| IM-MoCo          | 30.1 | 0.98 |

## Citation

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
