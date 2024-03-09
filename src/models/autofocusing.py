import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import FFT, IFFT


class Autofocusing(nn.Module):
    
      
    def __init__(self, masks):
        super().__init__()
      
        self.num_movements = masks.shape[0]
        self.motion_parameters = nn.ParameterDict(dict(
            rot_vector=torch.nn.Parameter(data=torch.zeros(self.num_movements )),
            x_shifts=torch.nn.Parameter(data=torch.zeros(self.num_movements )),
            y_shifts=torch.nn.Parameter(data=torch.zeros(self.num_movements )),
        ))
        
        self.device = masks.device
        self.masks = masks
      
    def forward(self, ks_input):
        x, num_lines = ks_input.shape
        
        num_movements = self.masks.shape[0]        
        images = IFFT(ks_input.squeeze().unsqueeze(0) * self.masks.float()).unsqueeze(1)
    
    
        angle = torch.deg2rad(self.motion_parameters['rot_vector'])
        rotation_matrix = torch.zeros((angle.shape[0], 2, 2), device= self.device)
        
        rotation_matrix[:, 0, 0] = rotation_matrix[:, 0, 0] + torch.cos(angle)
        rotation_matrix[:, 0, 1] = rotation_matrix[:, 0, 1] + -torch.sin(angle)
        rotation_matrix[:, 1, 0] = rotation_matrix[:, 1, 0] + torch.sin(angle)
        rotation_matrix[:, 1, 1] = rotation_matrix[:, 1, 1] + torch.cos(angle)
        
        rotation_matrix = rotation_matrix.permute(0, 2, 1)
        # correct for the shift in the center of rotation with matrix multiplication
        
        translations = torch.stack([self.motion_parameters['x_shifts'], self.motion_parameters['y_shifts']], dim=-1)
        
        shift = torch.zeros((num_movements, 2), device=ks_input.device)
        
        shift[:,0] =  shift[:,0] + (- rotation_matrix[:,0,0] * translations[:,0] - rotation_matrix[:,0,1] * translations[:,1])
        shift[:,1] = shift[:,0] + (- rotation_matrix[:,1,0] * translations[:,0] - rotation_matrix[:,1,1] * translations[:,1])
                
        torch_affine = torch.zeros((num_movements, 2, 3)).to(images.device)
        
        torch_affine[:,0,-1] = torch_affine[:,0,-1] + shift[:,0].float() 
        torch_affine[:,1,-1] = torch_affine[:,1,-1] + shift[:,1].float()
        torch_affine[:,0,0] = torch_affine[:,0,0] + rotation_matrix[:,0,0]
        torch_affine[:,0,1] = torch_affine[:,0,1] + rotation_matrix[:,0,1]   
        torch_affine[:,1,0] = torch_affine[:,1,0] + rotation_matrix[:,1,0]
        torch_affine[:,1,1] = torch_affine[:,1,1] + rotation_matrix[:,1,1]

        # normalize the translation to the to image size between -1 and 1
        torch_affine[:, :, -1] = torch_affine[:, :, -1] / ((torch.tensor(images[0,0,...].shape).to(images.device) * 2.) - 1 )


                
        grid = F.affine_grid(torch_affine, (self.num_movements, 2, x, num_lines), align_corners=True).to(images.device)
       
        image_2d = torch.view_as_complex(F.grid_sample(  
                torch.view_as_real(images.squeeze(1)).permute(0,3,1,2) , grid.float(), mode='bicubic', align_corners=False).squeeze(1).permute(0,2,3,1).contiguous())

       
        kspace_out = (ks_input.squeeze() * (1 - self.masks.sum(0)).float()) + ( FFT(image_2d) * self.masks.float()).sum(0)

        
        return kspace_out