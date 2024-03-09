import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import sys

sys.path.append('src/')

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from models.immoco import imcoco_motion_correction
from models.KLineDetect import get_unet
from utils.data_utils import FFT, IFFT
from utils.evaluate import calmetric2D
from utils.motion_utils import extract_movement_groups

# load model
net = get_unet(in_chans=2, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0).cuda()
net.load_state_dict(torch.load('src/model_weights/KLineDetect_4200_epochs_v2.pth'))

if os.path.exists('../results') == False:
    os.mkdir('../results')

scenarioius = ['light', 'heavy']
metrics_all = []

    
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
    net.eval()

    metrics = []
    for idx_test in trange(batch_test):
        #try:
        k_space_test = kspaces_test[idx_test].view(bsz,channels,H,W).cuda()
        img_motion_test = IFFT(k_space_test).abs()
    
        with torch.no_grad():
            mask = (net(torch.view_as_real(k_space_test / img_motion_test.std()).squeeze(1).permute(0,3,1,2)).sigmoid() > 0.5)
            masks = extract_movement_groups(mask.squeeze().sum(0).div(mask.squeeze().shape[0]) > 0.2, make_list=True).cuda()
        
        image_gt = images_gt[idx_test].abs().cuda()
        
        refined_image, _ = imcoco_motion_correction(k_space_test.squeeze(), masks, iters=200, learning_rate=1e-2, lambda_ge=1e-2, debug=False)
        
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
    
    fig.savefig('results/immoco_' + scenario + '.png', bbox_inches='tight')
    plt.close(fig)
    
    metrics_all.append(metrics)
    
# save metrics
torch.save(metrics_all, 'results/immoco_metrics.pth')

# load the metrics and loop over the list of lists of dictionaries, get the means and stds of each scenario for each metric, plot the violin plots and save them as a latex table
metrics_all = torch.load('results/immoco_metrics.pth')

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
with open('results/immoco_metrics.tex', 'w') as f:
    f.write('\\begin{table}[h]\n')
    f.write('\\centering\n')
    f.write('\\caption{immoco_metrics metrics}\n')
    f.write('\\label{tab:immoco_metrics}\n')
    f.write('\\begin{tabular}{l|cccc}\n')
    f.write('\\topline\n')
    f.write('Scenario & SSIM & PSNR & Haar psi & RMSE \\\\ \n')
    f.write('\\midline\n')
    for i in range(len(scenarioius)):
        f.write(scenarioius[i] + ' & ' + f'{means_all[i*4]:.2f}' + '$\pm$' + f'{stds_all[i*4]:.2f}' + ' & ' + f'{means_all[i*4+1]:.2f}' + '$\pm$' + f'{stds_all[i*4+1]:.2f}' + ' & ' + f'{means_all[i*4+2]:.2f}' + '$\pm$' + f'{stds_all[i*4+2]:.2f}' + ' & ' + f'{means_all[i*4+3]:.2f}' + '$\pm$' + f'{stds_all[i*4+3]:.2f}' + '\\\\ \n')
    f.write('\\bottomline\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{table}\n')
    
    
# # plot violin plots
# # # plot all metrics for metrics all moderate in one violin plot for each metric
# plt.style.use('seaborn-v0_8-paper')
# plt.rcParams.update({'font.size': 22, 'font.family': 'serif'})

# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
# axs[0].violinplot([metrics_for_plot[0]], positions=[1], showmeans=True, showmedians=True)
# axs[0].violinplot([metrics_for_plot[4]], positions=[2], showmeans=True, showmedians=True)
# axs[0].violinplot([metrics_for_plot[8]], positions=[3], showmeans=True, showmedians=True)

# axs[0].set_title('SSIM', fontsize=16)
# axs[0].set_xticks([1, 2, 3])
# axs[0].set_xticklabels(['light', 'severe'], fontsize=16)

# axs[1].violinplot([metrics_for_plot[1]], positions=[1], showmeans=True, showmedians=True)
# axs[1].violinplot([metrics_for_plot[5]], positions=[2], showmeans=True, showmedians=True)
# axs[1].violinplot([metrics_for_plot[9]], positions=[3], showmeans=True, showmedians=True)
# axs[1].set_title('PSNR', fontsize=16)
# axs[1].set_xticks([1, 2, 3])
# axs[1].set_xticklabels(['light', 'moderate', 'severe'], fontsize=16)

# axs[2].violinplot([metrics_for_plot[2]], positions=[1], showmeans=True, showmedians=True)
# axs[2].violinplot([metrics_for_plot[6]], positions=[2], showmeans=True, showmedians=True)
# axs[2].violinplot([metrics_for_plot[10]], positions=[3], showmeans=True, showmedians=True)

# axs[2].set_title('Haar Psi', fontsize=16)
# axs[2].set_xticks([1, 2, 3])
# axs[2].set_xticklabels(['light', 'moderate', 'severe'], fontsize=16)


# axs[3].violinplot([metrics_for_plot[3]], positions=[1], showmeans=True, showmedians=True)
# axs[3].violinplot([metrics_for_plot[7]], positions=[2], showmeans=True, showmedians=True)
# axs[3].violinplot([metrics_for_plot[11]], positions=[3], showmeans=True, showmedians=True)

# axs[3].set_title('RMSE', fontsize=16)
# axs[3].set_xticks([1, 2, 3])
# axs[3].set_xticklabels(['light', 'moderate', 'severe'], fontsize=16)


# # save figure
# fig.savefig('results/immoco_metrics.png', bbox_inches='tight')
# plt.close(fig)





