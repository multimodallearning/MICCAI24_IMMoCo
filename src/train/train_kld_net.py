import sys

sys.path.append('src/')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import trange

from src.models.kld_net import get_unet
from src.utils.data_utils import IFFT
from src.utils.evaluate import dice_coef, iou_coef, metrics_classification
from src.utils.motion_utils import motion_simulation2D

# initialize wandb
wandb.init(project="MICCAI24_MoCo", group="KLineDetect", name="KLineDetect", mode="online")


net = get_unet(in_chans=2, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0).cuda()

EPX = 4200
run_loss = torch.zeros(EPX)
val_loss = torch.zeros(EPX)

learning_rate = 1e-4

optimizer = torch.optim.AdamW(net.parameters(),lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,EPX//6,2)
torch.manual_seed(128)

bsz = 4
tst_bsz = 4


print('Loading data...')

data_train = torch.load('Dataset/Brain/t2/train_files/_train_data.pth')
data_val = torch.load('Dataset/Brain/t2/val_files/_val_data.pth')

k_space_train = data_train['kspace']
k_space_val = data_val['kspace']



batch_train, H, W = k_space_train.shape
batch_val,_,_ = k_space_val.shape

channels = 1

best_dice = 0.80
for i in trange(EPX):
    
    net.train()
    idx = torch.randperm(batch_train)[0:bsz]
    
    
    k_space = k_space_train[idx].view(bsz,channels,H,W).cuda()
    image_train = IFFT(k_space)
    
    mask = torch.zeros((bsz, H, W), dtype=torch.long).cuda()
    
    for trgt in range(bsz):
        k_space[trgt], mask[trgt], _, _ = motion_simulation2D(image_train[trgt].squeeze())
   
    k_space = torch.view_as_real(k_space.view(bsz,1,H,W)).squeeze(1).permute(0,3,1,2)
    
    preds = net(k_space)
   
    loss = criterion(preds.squeeze(), (mask.squeeze().float()))
    
    # log data to wandb
    wandb.log({"loss_train": loss, "epoch": i})
    
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
        
    if(i>5):
            scheduler.step()
            
    run_loss[i] = loss.item()
    
    net.eval()
    idx_test = torch.randperm(batch_val)[0:tst_bsz]
    
    with torch.no_grad():
       
        k_space_test = k_space_val[idx_test].view(tst_bsz,channels,H,W).cuda()
        image_val = IFFT(k_space_test)
        mask_test = torch.zeros((bsz, 1, H, W), dtype=torch.long).cuda()
        for trgt in range(tst_bsz):
            k_space_test[trgt], mask_test[trgt], _ ,_ = motion_simulation2D(image_val[trgt].squeeze())

        k_space_test = torch.view_as_real(k_space_test.view(tst_bsz,1,H,W)).squeeze(1).permute(0,3,1,2)
        
        preds_test =  net(k_space_test)
      
        output_test = (preds_test.squeeze())
       
        test_loss = criterion(output_test.squeeze(), (mask_test.squeeze().float()))
        
        wandb.log({"loss_val": test_loss, "epoch": i})
        
        dice_score = dice_coef(output_test.sigmoid() > 0.5, mask_test.squeeze().float())
        iou = iou_coef(output_test.sigmoid() > 0.5, mask_test.squeeze().float())
        calassification_metrics = metrics_classification((output_test.sigmoid() > 0.5).float(), mask_test.squeeze().float())
        
        wandb.log({"dice_score": dice_score, "epoch": i})
        wandb.log({"iou": iou, "epoch": i})
        # log pixelwise accuracy
        wandb.log({"pixelwise_accuracy": torch.sum((output_test.sigmoid() > 0.5) == mask_test.squeeze().float()) / (H*W), "epoch": i})

        wandb.log({"sensitivity": calassification_metrics['Sensistivity'], "epoch": i})
        wandb.log({"specificity": calassification_metrics['Specificity'], "epoch": i})
        wandb.log({"presission": calassification_metrics['Presission'], "epoch": i})
        wandb.log({"f1": calassification_metrics['F1'], "epoch": i})
        
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save(net.state_dict(), f'src/model_weights/kLDNet.pth')
            print(f'Current dice score: {dice_score:.4f}')
            print("Saved model to file")
        
        
    val_loss[i] = test_loss.item()

plt.plot(F.avg_pool1d(F.avg_pool1d(run_loss.view(1,1,-1),15,stride=3),15,stride=1).squeeze())
plt.plot(F.avg_pool1d(F.avg_pool1d(val_loss.view(1,1,-1),15,stride=3),15,stride=1).squeeze())
plt.legend(['train','val'])
plt.savefig(f'resutls/klDNet/lossWKLineDetect_{EPX}_epochs.png')

