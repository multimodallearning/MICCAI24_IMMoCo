import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fftn, fftshift, ifftn, ifftshift

from utils.pytorch_nufft import nufft

Ft = lambda x : fftshift(fftn(ifftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
IFt = lambda x : ifftshift(ifftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

# Algorithm
def get_rot_mat_nufft(rot_vector):
    rot_mat = torch.zeros(rot_vector.shape[0], 2, 2).cuda()
    rot_mat[:, 0, 0] = torch.cos(rot_vector)
    rot_mat[:, 0, 1] = -torch.sin(rot_vector)
    rot_mat[:, 1, 0] = torch.sin(rot_vector)
    rot_mat[:, 1, 1] = torch.cos(rot_vector)
    return rot_mat


def R_differentiable(ks, rot_vector, oversamp=5):
    rot_matrices = get_rot_mat_nufft(rot_vector)
    grid = torch.stack([
        arr.flatten() for arr in torch.meshgrid(
            torch.arange(-ks.shape[0]//2, ks.shape[0]//2).float(),
            torch.arange(-ks.shape[1]//2, ks.shape[1]//2).float(),
            indexing='ij')]).cuda()
    grid = (rot_matrices @ \
            grid.reshape(2, 320, 320).movedim(1, 0)).movedim(0, 1).reshape(2, -1)
    img = nufft.nufft_adjoint(ks, grid.T, device='cuda', oversamp=oversamp, 
                              out_shape=[1, 1, *ks.shape])[0, 0]
    return Ft(img)

def AFPLUS(ks, model):
    
    beta1, beta2 = 0.89, 0.8999
    ps = ks.shape[-1]
    ps_cf = int(ps // 2 * 0.08)
    zero_middle = torch.ones((ps)).cuda()
    zero_middle[ps // 2 - ps_cf : ps // 2 + ps_cf] = 0.
    img = IFt(ks).abs()

    x_shifts = torch.zeros(ps)
    y_shifts = torch.zeros(ps)
    x_shifts = torch.nn.Parameter(data=x_shifts.cuda(), requires_grad=True)
    y_shifts = torch.nn.Parameter(data=y_shifts.cuda(), requires_grad=True)
    x_moment1 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
    x_moment2 = torch.nn.Parameter(data=torch.zeros_like(x_shifts), requires_grad=True)
    y_moment1 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
    y_moment2 = torch.nn.Parameter(data=torch.zeros_like(y_shifts), requires_grad=True)
    rot_vector = torch.zeros(ps).cuda()
    rot_vector = torch.nn.Parameter(data=rot_vector.cuda(), requires_grad=True)
    rot_moment1 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)
    rot_moment2 = torch.nn.Parameter(data=torch.zeros_like(rot_vector), requires_grad=True)

    for _ in range(80):
        rot_vector = rot_vector * zero_middle
        x_shifts = x_shifts * zero_middle
        y_shifts = y_shifts * zero_middle

        # Translation
        phase_shift = -2 * math.pi * (
            x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
            y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
        new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + \
                                               phase_shift)).exp()
        # Rotation
        yp_ks = R_differentiable(new_k_space, rot_vector)
        yp_img = IFt(yp_ks).abs()

        loss_net = (yp_img[None, None] * 1e4 * model(yp_img[None, None] * 1e4).sigmoid()).mean()
        x_grad, y_grad, rot_grad = torch.autograd.grad(loss_net,
                                                       [x_shifts, y_shifts, rot_vector], create_graph=False)
        x_grad, y_grad = x_grad * 1e-4, y_grad * 1e-4
        rot_grad = rot_grad * 1e-4
        x_moment1 = beta1 * x_moment1.detach() + (1. - beta1) * x_grad
        x_moment2 = beta2 * x_moment2.detach() + (1. - beta2) * x_grad * x_grad + 1e-24
        y_moment1 = beta1 * y_moment1.detach() + (1. - beta1) * y_grad
        y_moment2 = beta2 * y_moment2.detach() + (1. - beta2) * y_grad * y_grad + 1e-24
        rot_moment1 = beta1 * rot_moment1 + (1. - beta1) * rot_grad
        rot_moment2 = beta2 * rot_moment2 + (1. - beta2) * rot_grad * rot_grad + 1e-24
        x_shifts = x_shifts - 3e-4 * x_moment1 * x_moment2.rsqrt()
        y_shifts = y_shifts - 3e-4 * y_moment1 * y_moment2.rsqrt()
        rot_vector = rot_vector - 3e-4 * rot_moment1 * rot_moment2.rsqrt()

    rot_vector = rot_vector * zero_middle
    x_shifts = x_shifts * zero_middle
    y_shifts = y_shifts * zero_middle
    # Translation
    phase_shift = -2 * math.pi * (
        x_shifts * torch.linspace(0, ps, ps)[None, :, None].cuda() + 
        y_shifts * torch.linspace(0, ps, ps)[None, None, :].cuda())[0]
    new_k_space = ks.abs().cuda() * (1j * (ks.angle().cuda() + \
                                           phase_shift)).exp()
    # Rotation
    yp_ks = R_differentiable(new_k_space, rot_vector)
    return IFt(yp_ks)