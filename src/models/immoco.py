import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output, display
from matplotlib import pyplot as plt

from utils.data_utils import FFT, IFFT
from utils.losses import GradientEntropyLoss

network_config = {
    "otype": "CutLassMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 256,
    "n_hidden_layers": 1,
}

mot_network_config = {
    "otype": "FullyFusedMLP",
    "activation": "Tanh",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 1,
}

encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "fine_resolution": 320,
    "per_level_scale": 2,
    "interpolation": "Linear",
}


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


def make_grids(sizes, device="cpu"):
    dims = len(sizes)
    lisnapces = [torch.linspace(-1, 1, s, device=device) for s in sizes]
    mehses = torch.meshgrid(*lisnapces, indexing="ij")
    coords = torch.stack(mehses, dim=-1).view(-1, dims)
    return coords


class IMMoCo(nn.Module):
    def __init__(self, masks):
        super().__init__()

        self.image_inr = tcnn.NetworkWithInputEncoding(
            2, 2, encoding_config, network_config
        )
        self.motion_inr = tcnn.NetworkWithInputEncoding(
            3, 2, encoding_config, mot_network_config
        )

        self.masks = masks
        self.num_movements, self.x, self.num_lines = masks.shape

        self.device = masks.device

        self.identy_grid = F.affine_grid(
            torch.eye(2, 3, device=self.masks.device).unsqueeze(0),
            torch.Size((1, 1, self.x, self.num_lines)),
            align_corners=True,
        )

        self.input_grid = make_grids(
            (self.num_movements, self.x, self.num_lines), device=self.device
        )

    def forward(self):

        image_prior = (
            self.image_inr(self.identy_grid.view(-1, 2))
            .float()
            .view(self.x, self.num_lines, 2)
        )
        image_prior = image_prior[..., 0] + 1j * image_prior[..., 1]

        images = image_prior.squeeze().unsqueeze(0).repeat(self.num_movements, 1, 1)

        grids = self.motion_inr(self.input_grid).float().tanh().view(
            self.num_movements, self.x, self.num_lines, 2
        ) + self.identy_grid.view(1, self.x, self.num_lines, 2)

        motion_images = torch.view_as_complex(
            F.grid_sample(
                torch.view_as_real(images).permute(0, 3, 1, 2),
                grids,
                mode="bilinear",
                align_corners=False,
                padding_mode="zeros",
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        kspace_out = (FFT(image_prior).squeeze() * (1 - self.masks.sum(0)).float()) + (
            FFT(motion_images) * self.masks.float()
        ).sum(0)

        return kspace_out, image_prior


def imcoco_motion_correction(
    kspace_corr, masks, iters=200, learning_rate=1e-2, lambda_ge=1e-2, debug=False
):
    """_summary_

    Args:
        kspace_corr (_type_): _description_
        masks (_type_): _description_
        iters (int, optional): _description_. Defaults to 200.
        learning_rate (_type_, optional): _description_. Defaults to 1e-2.
        lambda_ge (_type_, optional): _description_. Defaults to 1e-2.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # with ClearCache():

    IMOCO = IMMoCo(masks)

    # data should be between 1 and 16000
    scale = kspace_corr.abs().max()

    kspace_motion_norm = kspace_corr.div(scale).mul(16000)

    kspace_input = kspace_motion_norm.clone().detach().cuda()

    if debug:
        print(f"Scale: {scale:.4f}")
        print(
            f"Kspace input: {kspace_input.abs().min().item():.4f}, {kspace_input.abs().max().item():.4f}"
        )

    optimizer = torch.optim.Adam(
        [
            {"params": IMOCO.motion_inr.parameters(), "lr": learning_rate},
            {"params": IMOCO.image_inr.parameters(), "lr": learning_rate},
        ]
    )

    if debug:
        stats = []
        summar_steps = 20
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[2].set_title("Loss")
        axs[2].set_xlabel("Iterations")
        axs[2].set_xlim(0, iters)

    for j in range(iters):

        optimizer.zero_grad()

        kspace_foward_model, image_prior = IMOCO()

        loss_inr = F.mse_loss(
            torch.view_as_real(kspace_foward_model), torch.view_as_real(kspace_input)
        ) + GradientEntropyLoss()(image_prior).mul(lambda_ge)

        loss_inr.backward()
        optimizer.step()

        if debug:
            stats.append(loss_inr.item())

        if j % (iters // 10) and j > (iters // 2):
            lambda_ge *= 0.5
        if debug:
            if j % summar_steps == 0 or j == 199:
                print(f"iter: {j}, DC_Loss: {loss_inr:.4f}")

                axs[0].imshow(image_prior.abs().detach().cpu().squeeze(), cmap="gray")
                axs[0].set_title("IM-MoCo Image")
                axs[0].set_axis_off()

                axs[1].imshow(
                    IFFT(kspace_foward_model.detach().cpu()).abs(), cmap="gray"
                )
                axs[1].set_title("Motion Forward")
                axs[1].set_axis_off()

                axs[2].plot(stats)

                plt.close()  ##could prevent memory leak?
                display(fig)
                clear_output(wait=True)

    # clear all gpu memory
    torch.cuda.empty_cache()
    del kspace_input, kspace_motion_norm, IMOCO, optimizer, loss_inr

    return image_prior, kspace_foward_model
