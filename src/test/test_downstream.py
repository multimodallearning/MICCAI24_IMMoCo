import os
import sys

sys.path.append('src/')
from collections import defaultdict

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from matplotlib import pyplot as plt
from torchvision.ops import box_convert

from models.KLineDetect import get_unet
from utils.data_utils import FFT, IFFT
from utils.motion_utils import extract_movement_groups


class GradientEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
         
    def entropy(self, x):
        return -torch.sum(torch.mul(x , torch.log(x + 1e-8)))
    
    def forward(self, x):
        
        #loss = 0
        #for x in [x.real, x.imag]:
        dx = (x[:, :-1] - x[ :, 1:]).abs()
        dy = ((x[:-1, :] - x[ 1:, :])).abs()
        
        # pad the gradient
        dx = F.pad(dx, (0, 1, 0, 0), mode='constant', value=0)
        dy = F.pad(dy, (0, 0, 0, 1), mode='constant', value=0)
        
        gradient = dx + dy
        
        loss = self.entropy(gradient)
  
        return loss
    
def extract_patches(images, points, patch_size=32):
    """
    Extract patches from a point cloud
    :param images: torch.Tensor (B, C, H, W)
    :param points: torch.Tensor (B, N, 3)
    :param patch_size: int
    :return: torch.Tensor (B*patches, C, patch_size, patch_size)
    """

    size = (1,1,patch_size,patch_size)
    grid = (F.affine_grid(torch.eye(2,3).unsqueeze(0)*.2,size=size, align_corners=False).view(1,1,-1,2) + points.unsqueeze(0).unsqueeze(2)).to(images.device)
    
    patches = F.grid_sample(images.float(), grid.float(), align_corners=True).view(-1, images.shape[1], patch_size, patch_size)

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
            
    patches_1 = extract_patches(image1[None, None], torch.stack(boxes), patch_size=124)
    patches_2 = extract_patches(image2[None, None], torch.stack(boxes), patch_size=124)

    for i in range(patches_1.shape[0]):
       
        psnr, ssim, haarpsi, rmse = calmetric2D(patches_1[i][None], patches_2[i][None])
        
        psnrs.append(psnr)
        ssims.append(ssim)
        rmses.append(rmse)
        haarpsis.append(haarpsi)
        
    metrics['ssim'] = torch.mean(torch.tensor(ssims))
    metrics['psnr'] = torch.mean(torch.tensor(psnrs))
    metrics['haarpsi'] = torch.mean(torch.tensor(haarpsis))
    metrics['rmse'] = torch.mean(torch.tensor(rmses))
    
    return metrics

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        
import tinycudann as tcnn

network_config={
    "otype": "FullyFusedMLP",
    "activation": 'ReLU',
    "output_activation": 'None',
    "n_neurons": 64,
    "n_hidden_layers": 1,
    "dtype": "float32"
}

mot_network_config={
    "otype": "FullyFusedMLP",
    "activation": 'Tanh',
    "output_activation": 'None',
    "n_neurons": 16,
    "n_hidden_layers": 1,
    "dtype": "float32"
}

encoding_config = {
    'otype': 'Grid',
    'type': 'Hash',
    'n_levels': 16,
    'n_features_per_level': 2,
    'log2_hashmap_size': 19,
    'base_resolution': 16,
    'fine_resolution': 320,
    'per_level_scale': 2,
    'interpolation': 'Linear' }

def make_grids(sizes, device='cpu'):
    dims = len(sizes)
    lisnapces = [torch.linspace(-1,1,s, device=device) for s in sizes]
    mehses = torch.meshgrid(*lisnapces, indexing='ij')
    coords = torch.stack(mehses,dim=-1).view(-1,dims)
    return coords

class IMMoCo(nn.Module):
    def __init__(self, masks):
        super().__init__()
        
        self.image_inr = tcnn.NetworkWithInputEncoding(2, 2, encoding_config, network_config)
        self.motion_inr = tcnn.NetworkWithInputEncoding(3, 2, encoding_config, mot_network_config)
        
        self.masks = masks
        self.num_movements, self.x, self.num_lines = masks.shape
        
        self.device = masks.device
        
        self.identy_grid = F.affine_grid(torch.eye(2,3, device=self.masks.device).unsqueeze(0), torch.Size((1, 1, self.x, self.num_lines)), align_corners=True)
        
        self.input_grid = make_grids((self.num_movements, self.x, self.num_lines), device=self.device)
        
    def forward(self):
        
        image_prior = self.image_inr(self.identy_grid.view(-1,2)).float().view(self.x, self.num_lines, 2)
        image_prior = image_prior[...,0 ] + 1j * image_prior[...,1]
        
        images = image_prior.squeeze().unsqueeze(0).repeat(self.num_movements,1,1)
        
        grids = self.motion_inr(self.input_grid).float().tanh().view(self.num_movements, self.x, self.num_lines, 2) + self.identy_grid.view(1, self.x, self.num_lines, 2)
        
        motion_images = torch.view_as_complex(F.grid_sample(
                torch.view_as_real(images).permute(0,3,1,2).contiguous() , grids , mode='bilinear', align_corners=False, padding_mode='zeros').contiguous().permute(0,2,3,1).contiguous())
        
        kspace_out = (FFT(image_prior).squeeze() * (1 - self.masks.sum(0)).float()) + (FFT(motion_images) * self.masks.float()).sum(0)
        
        return kspace_out, image_prior



def imcoco_motion_correction(kspace_corr, masks, iters=200, learning_rate=1e-2, lambda_ge=1e-2, debug=False):
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
    with ClearCache():
        
        IMOCO = IMMoCo(masks)

        # data should be between 1 and 16000
        scale = kspace_corr.abs().max()
    
        kspace_motion_norm = kspace_corr.div(scale) * 8000
        
        kspace_input = kspace_motion_norm.clone().detach().cuda()
        
        if debug:
            print(f'Scale: {scale:.4f}')
            print(f'Kspace input: {kspace_input.abs().max():.4f}')
            
        optimizer = torch.optim.Adam([{'params': IMOCO.motion_inr.parameters(), 'lr': learning_rate}, {'params': IMOCO.image_inr.parameters(), 'lr': learning_rate}])
        
        if debug:
            stats = []          
            summar_steps = 20

        for j in range(iters):
        
            optimizer.zero_grad()
            
            kspace_foward_model, image_prior = IMOCO()

            loss_inr = F.mse_loss(torch.view_as_real(kspace_foward_model), torch.view_as_real(kspace_input)) + GradientEntropyLoss()(image_prior).mul(lambda_ge)

            loss_inr.backward()
            optimizer.step()
            
            if debug:
                stats.append(loss_inr.item())
            
            if j % 10 ==0 and j > 80:
                lambda_ge *= 0.5
            if debug:
                if j % summar_steps == 0 or j == 199:
                    print(f'iter: {j}, DC_Loss: {loss_inr:.4f}')
   
    # clear all gpu memory
    torch.cuda.empty_cache()
    del kspace_input, kspace_motion_norm, IMOCO, optimizer, loss_inr
    
    return image_prior, kspace_foward_model


def coords_to_pixel(coords, im_size):
    
    """
    Convert coordinates from bounding box format to image format
    """
    x, y, w, h = coords
    x = int(x * im_size[1])
    y = int(y * im_size[0])
    w = int(w * im_size[1])
    h = int(h * im_size[0])
    return x, y, w, h

def motion_test_detection_data(path, annotations_path):
    
    # load model
    net = get_unet(in_chans=2, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0).cuda()
    net.load_state_dict(torch.load('src/model_weights/KLineDetect_4200_epochs_v2.pth'))

    bboxes = []
    size = 640
    
    file_names = np.sort(os.listdir(path))
    
    for annotation_file in os.listdir(annotations_path):
        with open(os.path.join(annotations_path, annotation_file), 'r') as f:
            lines = f.readlines()
            boxes_gt = []
            for line in lines:
                line = line.strip().split()
                # Format: ['', class, center_x, center_y, width, height],  remove the first element
                # class_id, center_x, center_y, width, height = map(float, line[1:])
                class_id, center_x, center_y, width, height = map(float, line)
                center_x = center_x *2 - 1
                center_y = center_y *2 - 1
                
               # center_x, center_y, width, height = box_convert(torch.tensor(coords_to_pixel([center_x, center_y, width, height], (size, size))), 'cxcywh', 'xyxy')
                boxes_gt.append(torch.tensor([center_x, center_y]))
            # check if tensorlist is empty
            if len(boxes_gt) > 0:
                bboxes.append(boxes_gt)
            f.close() 
            
    if os.path.exists('results') == False:
        os.mkdir('results')

    scenarioius = ['light', 'heavy']
    metrics_all = defaultdict(list)

    for scenario in scenarioius:
        print('Scenario: ', scenario)
        data_path = 'Dataset/DetectionData/test/images_' + scenario + '/_detection_test_data.pth'
        
        output_path = data_path.split('/_detection')[0] + '_corrected'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        data_test = torch.load(data_path)

        kspaces_test = data_test['kspace_motion']
        images_gt = data_test['image_rss']
        
        metrics = []
        batch_test, H, W = kspaces_test.shape
        
        bsz = 1
        channels = 1
        net.eval()
        motion_image = []
        moco_image = []
        gt_image = []
        metrics = []
        for idx_test, file_name in zip(range(batch_test), file_names):
            if not file_name.endswith('.png'):
                continue    
            print('Processing: ', file_name)
            #try:
            k_space_test = kspaces_test[idx_test].view(bsz,channels,H,W).cuda()
            img_motion_test = IFFT(k_space_test).abs()
            motion_image.append(img_motion_test.detach().cpu())
            with torch.no_grad():
                mask = (net(torch.view_as_real(k_space_test / img_motion_test.std()).squeeze(1).permute(0,3,1,2)).sigmoid() > 0.5)#.float()
                masks = extract_movement_groups(mask.squeeze().sum(0).div(mask.squeeze().shape[0]) > 0.2, make_list=True).cuda()

            image_gt = images_gt[idx_test].abs().cuda()
            
            gt_image.append(image_gt.detach().cpu())
            
            image_prior, motion_out = imcoco_motion_correction(k_space_test.squeeze(), masks, iters=200, learning_rate=1e-2, lambda_ge=1e-2, debug=False)
            
            moco_image.append(image_prior.detach().cpu())
           
            patch_metrics = evaluate_patches(image_prior.detach().abs(), image_gt.squeeze(), bboxes[idx_test])
            
            # make a dictionary with the metrics
            metrics.append({'ssim': patch_metrics['ssim'],
                            'psnr': patch_metrics['psnr'],
                            'haar_psi': patch_metrics['haarpsi'],
                            'rmse': patch_metrics['rmse']})
            # normalize the image
            motion_corr = image_prior.detach().cpu().abs()
            motion_corr = (motion_corr - motion_corr.min())/(motion_corr.max() - motion_corr.min()) * 255.0
            torchvision.io.write_png(motion_corr.to(torch.uint8).unsqueeze(0), os.path.join(output_path, file_name))
            #except:
            #    print('Error in file: ', file_name)
        data = {'gt_image': torch.stack(gt_image),
                'moco_image': torch.stack(moco_image),
                'motion_image': torch.stack(motion_image),
                'metrics': metrics}
        
        # save the dict as torch file
        torch.save(data, output_path + '/_detection_test_data_corrected.pth')
        # append the metrics to the metrics_all list
        metrics_all[scenario] = metrics  
        
    return data, metrics_all 

scenarios = ['light', 'heavy']
data, metrics_all = motion_test_detection_data('Dataset/DetectionData/test/images', 'Dataset/DetectionData/test/labels')

