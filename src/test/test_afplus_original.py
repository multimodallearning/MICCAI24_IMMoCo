import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

sys.path.append('src/')

import matplotlib.pyplot as plt
import torch
from torch.fft import fftn, fftshift, ifftn, ifftshift
from tqdm import trange

from models.af_plus import AFPLUS
from utils.evaluate import calmetric2D

Ft = lambda x : fftshift(fftn(ifftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
IFt = lambda x : ifftshift(ifftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))


if os.path.exists('results') == False:
    os.mkdir('results')

scenarioius = ['light', 'heavy']
metrics_all = []

from models.unet import Unet

unet = Unet(1, 1, 32, 6, batchnorm=torch.nn.InstanceNorm2d, init_type='none').cuda()
unet.load_state_dict(torch.load('src/model_weights/AFPlus_best.pth'))
unet.eval()
    
for scenario in scenarioius:
    print('Loading data...')
    data_path = 'Dataset/Brain/t2/test_files/_test_data_' + scenario + '.pth'
    data_test = torch.load(data_path)

    kspaces_test = data_test['kspace_motion']
    images_gt = data_test['image_rss']
    
    metrics = []
    batch_test, H, W = kspaces_test.shape
    
    bsz = 1
    channels = 1

    metrics = []
    for idx_test in trange(batch_test):
        #try:
        k_space_test = kspaces_test[idx_test].view(bsz,channels,H,W).cuda()
        img_motion_test = IFt(k_space_test).abs()
    
        image_gt = images_gt[idx_test].abs().cuda()
        
        refined_image = AFPLUS(k_space_test.squeeze(), unet)
        
        with torch.no_grad():   
            motion_refined_test = refined_image.abs()
        
        crop = [int(H/4), int(W/4)]
        image_gt_crop = image_gt[crop[0]:-crop[0], crop[1]:-crop[1]].abs()
        motion_refined_test_crop = motion_refined_test[crop[0]:-crop[0], crop[1]:-crop[1]]
        psnr_test, ssim_test, haar_psi_test, rmse_test = calmetric2D(motion_refined_test_crop.unsqueeze(0).unsqueeze(0), image_gt_crop.unsqueeze(0).unsqueeze(0))
        # psnr_array, ssim_array, haar_psi_array, rmse_array
        #psnr_test, ssim_test, haar_psi_test, rmse_test = calmetric2D(motion_refined_test.unsqueeze(0).unsqueeze(0), image_gt.unsqueeze(0).unsqueeze(0))
                
        metrics.append({'ssim': ssim_test, 'psnr': psnr_test, 'haar_psi': haar_psi_test, 'rmse': rmse_test})
            
        # except Exception as e:
        #     print(e)
        #     print('Error in scenario ' + scenario + ' and image ' + str(idx_test) + ', skipping...')
        #     continue
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].imshow(img_motion_test.squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title('Motion corrupted image', fontsize=16)
    axs[1].imshow(motion_refined_test.squeeze().detach().cpu().numpy(), cmap='gray')
    axs[1].set_title('Motion corrected image', fontsize=16)
    axs[2].imshow(image_gt.squeeze().cpu().numpy(), cmap='gray')
    axs[2].set_title('Ground truth image', fontsize=16)
    
    fig.savefig('results/afplus_orig_' + scenario + '_v2.png', bbox_inches='tight')
    plt.close(fig)
    
    metrics_all.append(metrics)
    
# save metrics
torch.save(metrics_all, 'results/afplus_orig_metrics_v2.pth')

# load the metrics and loop over the list of lists of dictionaries, get the means and stds of each scenario for each metric, plot the violin plots and save them as a latex table
metrics_all = torch.load('results/afplus_orig_metrics_v2.pth')

metrics_for_plot = []
means_all = []
stds_all = []

for scenario in range(len(metrics_all)):
    
    for metric in metrics_all[scenario][0].keys():
       
        metrics = []
        for i in range(len(metrics_all[scenario])):
            metrics.append(metrics_all[scenario][i][metric])
        
        metrics_for_plot.append(torch.tensor(metrics))
        means_all.append(torch.mean(torch.tensor(metrics)))
        stds_all.append(torch.std(torch.tensor(metrics)))

metrics_for_plot = torch.stack(metrics_for_plot).squeeze().view(8, 50)
means_all = torch.stack(means_all)
stds_all = torch.stack(stds_all)

# save means +- stds in a latex table
print('Saving latex table...')
with open('results/afplus_orig_metrics_v2.tex', 'w') as f:
    f.write('\\begin{table}[h]\n')
    f.write('\\centering\n')
    f.write('\\caption{afplus_orig_metrics metrics}\n')
    f.write('\\label{tab:afplus_orig_metrics}\n')
    f.write('\\begin{tabular}{l|cccc}\n')
    f.write('\\topline\n')
    f.write('Scenario & SSIM & PSNR & Haar psi & RMSE \\\\ \n')
    f.write('\\midline\n')
    for i in range(len(scenarioius)):
        f.write(scenarioius[i] + ' & ' + f'{means_all[i*4]:.2f}' + '$\pm$' + f'{stds_all[i*4]:.2f}' + ' & ' + f'{means_all[i*4+1]:.2f}' + '$\pm$' + f'{stds_all[i*4+1]:.2f}' + ' & ' + f'{means_all[i*4+2]:.2f}' + '$\pm$' + f'{stds_all[i*4+2]:.2f}' + ' & ' + f'{means_all[i*4+3]:.2f}' + '$\pm$' + f'{stds_all[i*4+3]:.2f}' + '\\\\ \n')
    f.write('\\bottomline\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{table}\n')