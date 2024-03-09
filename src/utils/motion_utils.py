# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.9.12 ('pytorch')
#     language: python
#     name: python3
# ---
1
# %%
import numpy as np
import torch
import torch.nn.functional as F

from utils.data_utils import FFT, IFFT


# %%
def generate_list(size, n_movements, mingap=4, acs = 24):
    # Generate a list of random numbers that sum to size with a minimum gap of mingap and if movemement is in the center of the size then discard and sample again
    
    # slack = size - mingap * (n_movements - 1)
    
    # steps = torch.randint(0, slack, (1,))[0]

    # increments = torch.hstack([torch.ones((steps,), dtype=torch.long), torch.zeros((n_movements,), dtype=torch.long)])
    
    # increments = increments[torch.randperm(increments.shape[0])]

    # locs = torch.argwhere(increments == 0).flatten()
    # return (torch.cumsum(increments, dim=0)[locs] + mingap * torch.arange(0, n_movements))

    
    # generate a list of random numbers from 0 to size - 1 with length n_movements and a minimum gap of mingap between each number in the list but we don't sample in a region of 12 before and after the center 
    # size: number of phase lines
    # n_movements: number of movements
    # mingap: minimum gap between each movement
    
    # # Generate a range of indices excluding the center values from the ACS region
    indices =  np.hstack([np.arange(0, (size//2) - (acs//2), step=mingap), np.arange((size//2) + (acs//2), size, step=mingap)])
    # Sample indices from the remaining range
    sampled_indices = np.sort(np.random.choice(indices, n_movements, replace=False).astype(int))
    
    return sampled_indices

# %%
def add_noise(x, inputSnr):
    noiseNorm = torch.linalg.norm(x.view(-1).T) * 10 ** (-inputSnr / 20)
    noise = torch.randn_like(x)
    noise = noise / torch.linalg.norm(noise.view(-1).T) * noiseNorm
    y = x + noise
    return y
# %%
def get_rand_int(data_range, size=None):
    if size is None:
        rand = torch.randint(data_range[0], data_range[1], size=(1,))
        if rand == 0:
            rand = rand + 1
    else:
        rand = torch.randint(data_range[0], data_range[1], size=size)
    return rand
# %%
def extract_movements(indices):
    
    phase_lines = indices.shape[0]
    mask = torch.zeros((phase_lines, phase_lines), dtype=torch.long)
    count = 1
    for i in range(phase_lines):
        if i!=phase_lines - 1 and indices[i] == 1 and indices[i+1] == 1:
                mask[:,i] = count
        elif i!=phase_lines - 1  and indices[i] == 1 and indices[i+1] == 0:
                mask[:,i] = count
                count += 1
        elif i == phase_lines - 1 and indices[i] == 1:
                mask[:,i] = count
        else:
                pass 
    
    return mask


def extract_movement_groups(motionline_indcies, make_list=False):
    """_summary_

    Returns:
        torch.Tensor: motion_groups (phase_encoding, frequency_encoding) where each pixel is assigned a group number

    example:
    motion_groups = extract_movement_groups(motion_mask.squeeze().sum(0) // phase_encoding)

    num_movements = (motion_groups).unique().numel() - 1
    """
    phase_lines = motionline_indcies.shape[0]

    motion_groups = torch.zeros((phase_lines, phase_lines), dtype=torch.long, device=motionline_indcies.device)

    count = 1
    for phase_line in range(phase_lines):
        if (
            phase_line != phase_lines - 1
            and motionline_indcies[phase_line] == 1
            and motionline_indcies[phase_line + 1] == 1
        ):
            motion_groups[:, phase_line] = count
        elif (
            phase_line != phase_lines - 1
            and motionline_indcies[phase_line] == 1
            and motionline_indcies[phase_line + 1] == 0
        ):
            motion_groups[:, phase_line] = count
            count += 1
        elif phase_line == phase_lines - 1 and motionline_indcies[phase_line] == 1:
            motion_groups[:, phase_line] = count
        else:
            pass
        
    if make_list:
        unique = torch.unique(motion_groups).nonzero().squeeze()
       
        counts = unique.numel()
       
        motion_lists = torch.zeros((counts, *motion_groups.shape), dtype=torch.long, device=motionline_indcies.device)

        for i in range(counts):
            
            motion_lists[i, (motion_groups == i+1).bool()] = 1
        motion_groups = motion_lists
        
    return motion_groups

# method to expand transforms to number of lines with zeros depending on the mask
def extend_transforms(ks, masks, angles, x_shifts, y_shifts):
    H, W = ks.shape[-2:]
    new_angels = torch.zeros((ks.shape[0],)).to(angles.device)
    new_trans_x = torch.zeros((ks.shape[0],)).to(x_shifts.device)
    new_trans_y = torch.zeros((ks.shape[0],)).to(y_shifts.device)
    indecies = masks.sum(1) / H
     
    for i in range(masks.shape[0]):
        
        mask = indecies[i].to(angles.device)
        new_angels[mask == 1] = angles[i]
        new_trans_x[mask == 1] = x_shifts[i]
        new_trans_y[mask == 1] = y_shifts[i]
    return new_angels, new_trans_x, new_trans_y


def rotation_matrix(angle):
    """2D rotation matrix."""
    angle = torch.deg2rad(angle)
    rotation_matrix = torch.zeros((angle.shape[0], 2, 2), device=angle.device)
    rotation_matrix[:, 0, 0] = torch.cos(angle)
    rotation_matrix[:, 0, 1] = -torch.sin(angle)
    rotation_matrix[:, 1, 0] = torch.sin(angle)
    rotation_matrix[:, 1, 1] = torch.cos(angle)
    return rotation_matrix


# %%
def rotation_matrix_2d(angle, device=None):
    """2D rotation matrix."""
    angle = torch.deg2rad(angle)
    return torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                            [torch.sin(angle), torch.cos(angle)]], device=device)



# %%
def motion_simulation2D(image_2d, n_movements=None, snr=0, ):
    
        ksp_corrupt = FFT(image_2d)
        x, num_lines = ksp_corrupt.shape

        channels = 1 
        
        
        if n_movements is None:
                n_movements = get_rand_int([5, 20]).item()

        mingap = num_lines // n_movements
        
        acs = int(num_lines * 0.08) # 8% of the lines are ACS lines

        rand_list = generate_list(num_lines, n_movements, mingap, acs)
        motion_rng = [1,n_movements]

        mask = torch.zeros((x,num_lines), dtype=torch.long)
        rotations = torch.zeros((n_movements,))
        translations = torch.zeros((n_movements, 2))
        shift_x_rng = [[-10, 10]]  *   motion_rng[1]  
        shift_y_rng = [[-10, 10]]  *   motion_rng[1] 
        rotate_rng = [[-10, 10]]   *   motion_rng[1]  
        shift_w_rng = [[-2, 2]]  *   motion_rng[1]
        motion_list = [[]] * motion_rng[1]
        #w_rng = [10, mingap] 
        w_rng = [1, 10]

        for motion in range(n_movements):
                
                shift = [get_rand_int(shift_x_rng[motion]).item(), get_rand_int(shift_y_rng[motion]).item()]
                angle = get_rand_int(rotate_rng[motion])
                rotation = rotation_matrix_2d(angle)
                torch_affine = torch.tensor([[1, 0, shift[0]], [0, 1, shift[1]]]).float()
                torch_affine[:2, :2] = rotation
                torch_affine = torch_affine.view(1, 2, 3)

                torch_affine[:, :, -
                1] /= (torch.tensor(image_2d[0,...].shape)* 2.) - 1 
                
                grid = F.affine_grid(# type: ignore                        
                                                torch_affine, (1, channels, x, num_lines), align_corners=True).to(image_2d.device)
                image_2d_transformed_real = F.grid_sample(  # type: ignore
                        image_2d.unsqueeze(0).unsqueeze(0).real, grid.float(), mode='bilinear', padding_mode='border', align_corners=False)
                
                image_2d_transformed_imag = F.grid_sample(  # type: ignore
                        image_2d.unsqueeze(0).unsqueeze(0).imag, grid.float(), mode='bilinear', padding_mode='border', align_corners=False)
                
                image_2d_transformed = image_2d_transformed_real + 1j * image_2d_transformed_imag
                
                # ksp for shifted image
                ksp_shiftnrotate = FFT(image_2d_transformed).squeeze()
               
                # replace the ksp
                w_start = rand_list[motion]
                w_end = w_start + get_rand_int(w_rng)
                
                ksp_corrupt[..., w_start:w_end] = ksp_shiftnrotate[..., w_start:w_end]
                mask[:,w_start:w_end] = 1 
                motion_list.append([w_start.item(), w_end])
                
                rotations[motion] = angle
                translations[motion, :] = torch.tensor(shift)
        
        # add noise
        if snr > 0:
                ksp_corrupt = FFT(add_noise(IFFT(ksp_corrupt), snr))
                
       
        return ksp_corrupt, mask, rotations, translations
    
    
import torch.nn.functional as F


def rotation_matrix(angle):
    """2D rotation matrix."""
    angle = torch.deg2rad(angle)
    rotation_matrix = torch.zeros((angle.shape[0], 2, 2), device=angle.device)
    rotation_matrix[:, 0, 0] = rotation_matrix[:, 0, 0] + torch.cos(angle)
    rotation_matrix[:, 0, 1] = rotation_matrix[:, 0, 1] + -torch.sin(angle)
    rotation_matrix[:, 1, 0] = rotation_matrix[:, 1, 0] + torch.sin(angle)
    rotation_matrix[:, 1, 1] = rotation_matrix[:, 1, 1] + torch.cos(angle)
    return rotation_matrix

def motion_correction(kspace, mask, angles, translations, scale=False):
        x, num_lines = kspace.shape
        
        mask = mask.clone().detach().float().to(kspace.device).requires_grad_(True)
        
        num_movements = mask.shape[0]        
        images = IFFT(kspace.squeeze().unsqueeze(0) * mask.float()).unsqueeze(1)
        
        if scale:
                rotation = rotation_matrix(angles.tanh() * 2).permute(0, 2, 1)
        else:
                rotation = rotation_matrix(angles).permute(0, 2, 1)
       
        # correct for the shift in the center of rotation with matrix multiplication
        shift = torch.zeros((num_movements, 2), device=kspace.device)
        
        if scale:
                shift[:,0] = - rotation[:,0,0] * translations[:,0].tanh() * 5 - rotation[:,0,1] *  translations[:,1].tanh() * 5
                shift[:,1] = - rotation[:,1,0] * translations[:,0].tanh() * 5 - rotation[:,1,1] * translations[:,1].tanh() * 5
                
        else:
                shift[:,0] = - rotation[:,0,0] * translations[:,0] - rotation[:,0,1] * translations[:,1]
                shift[:,1] = - rotation[:,1,0] * translations[:,0] - rotation[:,1,1] * translations[:,1]
        
        torch_affine = torch.zeros((num_movements, 2, 3)).to(images.device)
        
        torch_affine[:,0,-1] = torch_affine[:,0,-1] + shift[:,0].float() 
        torch_affine[:,1,-1] = torch_affine[:,1,-1] + shift[:,1].float()
        torch_affine[:,0,0] = torch_affine[:,0,0] + rotation[:,0,0]
        torch_affine[:,0,1] = torch_affine[:,0,1] + rotation[:,0,1]   
        torch_affine[:,1,0] = torch_affine[:,1,0] + rotation[:,1,0]
        torch_affine[:,1,1] = torch_affine[:,1,1] + rotation[:,1,1]

        # normalize the translation to the to image size between -1 and 1
        torch_affine[:, :, -1] = torch_affine[:, :, -1] / ((torch.tensor(images[0,0,...].shape).to(images.device) * 2.) - 1 )
                
        grid = F.affine_grid(torch_affine, (num_movements, 2, x, num_lines), align_corners=True).to(images.device)
       
        image_2d = torch.view_as_complex(F.grid_sample(  
                torch.view_as_real(images.squeeze()).permute(0,3,1,2) , grid.float(), mode='nearest', align_corners=False).squeeze().permute(0,2,3,1).contiguous())

        # image_2d_transformed_imag = F.grid_sample( 
        #         images.imag, grid.float(), mode='nearest', align_corners=False).squeeze()

        #image_2d_transformed = image_2d_transformed_real + 1j * image_2d_transformed_imag
       
        kspace_out = (kspace.squeeze() * (1 - mask.sum(0)).float()) + ( FFT(image_2d) * mask.float()).sum(0)
        
        # print('kspace', kspace.requires_grad)
        # print('mask', mask.requires_grad)
        # print('images', images.requires_grad)
        # print('angles', angles.requires_grad)
        # print('rotation', rotation.requires_grad)
        # print('shift', shift.requires_grad)
        # print('torch_affine', torch_affine.requires_grad)
        # print('grid', grid.requires_grad)
        # print('torch_affine', torch_affine.requires_grad)
        # print('image_2d', image_2d.requires_grad)
        # print('kspace', kspace.requires_grad)
 

        return kspace_out