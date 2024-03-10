import h5py
import torch
from torch.fft import fftn, fftshift, ifftn, ifftshift


def load_file(path):
    """load h5 file."""
    with h5py.File(path, "r") as f:
        ksapce = f["kspace"][()]
    f.close()
    return ksapce


def prepare_data(kspaces, crop_size=320):
    """prepare data."""

    crop_size = 320
    image_gt = IFFT(kspaces)
    crop_x = image_gt.shape[-2] // 2 - crop_size // 2
    crop_y = image_gt.shape[-1] // 2 - crop_size // 2
    image_gt = image_gt[:, crop_x : crop_x + crop_size, crop_y : crop_y + crop_size]
    image_rss_comp = rss_comp(image_gt, dim=0)

    kspace = FFT(image_rss_comp)

    return kspace, image_rss_comp


def FFT(x):
    return fftshift(fftn(ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


def IFFT(x):
    return ifftshift(ifftn(fftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


def normalize_image(image):
    """normalize image 0 to 1."""

    image_abs = torch.abs(image)

    if (image_abs.max() - image_abs.min()) < 1e-12:
        return image - image_abs.min() + 1e-12
    else:
        return (image - image_abs.min()) / (image_abs.max() - image_abs.min())


def scale_image(image, scale=None):
    """scale image by scale."""
    if scale == None:
        return image / image.abs().max()
    return image / scale


def rescale_image(image, scale=None):
    """rescale image by scale."""
    return image * scale


def rss_comp(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    rss_real = torch.sqrt((data.real**2).sum(dim))
    rss_imag = torch.sqrt((data.imag**2).sum(dim))
    return rss_real + 1j * rss_imag
