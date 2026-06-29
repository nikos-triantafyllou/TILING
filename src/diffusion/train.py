import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--continue_training --checkpoint 256 --data1 'Tb' --data2 'galaxy_counts' --condition_on_NGPE 0", type=int, default=1)
parser.add_argument("--do_train", type=int, default=0)
parser.add_argument("--continue_training", type=int, default=256)
parser.add_argument("--checkpoint", type=int, default=256)
parser.add_argument("--data1", type=str, default='Tb')
parser.add_argument("--data2", type=str, default='galaxy_counts')
parser.add_argument("--condition_on_NGPE", type=int, default=0)
args = parser.parse_args()
data1 = str(args.data1)  #'Tb'  # 'Tb' or 'galaxy_counts'
data2 = str(args.data2)  #'galaxy_counts'  # 'Tb' or '
if data2 == 'None': observe = [data1]
else: observe = [data1, data2]
print('data1:', data1, 'data2:', data2, flush = True)

from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np 
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math 
from torch.nn.parallel import DataParallel as DP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from diff_tools import UNet
import diff_tools
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from datetime import datetime
from tqdm.auto import tqdm
import powerbox as pbox
print("Starting date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush = True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("GPU count:", torch.cuda.device_count(), flush=True)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None", flush=True)

datasetpath = '/leonardo_scratch/fast/IscrB_IC-DIFF/ntriantafyllou/sub_databases/personal_projects/ICs_project/v1/'
# NGPE_path = '/leonardo_scratch/fast/CNHPC_1497299/ntriantafyllou/ICs/'
NGPE_path =  '/leonardo_scratch/fast/INA24_C7B14/ntriantafyllou/ICs/'

datasetpath_ICs = '/leonardo_scratch/fast/IscrB_IC-DIFF/ntriantafyllou/sub_databases/personal_projects/ICs_project/v1/'


if data1 in ['Tb_RSDs_noise10_000', 'Tb_RSDs_noise10_000_wedge1', 'Tb_RSDs_noiseAA410_000', 'Tb_RSDs_noiseAA410_000_wedge1', 'Tb_RSDs_noise_iv_filter_1000_wedge1',
'Tb_RSDs_noise_no_filter_1000_wedge1','Tb_RSDs_noise_w_filter_1000_wedge1','Tb_RSDs_noise_w_filter_10_000_wedge1', 'Tb_RSDs_noise_iv_filter_1000', 'Tb_RSDs_noise_iv_filter_10_000_wedge1','Tb_RSDs_noise_iv_filter_100_wedge1']:
    datasetpath_data1 = '/leonardo_scratch/fast/CNHPC_1497299_0/ntriantafyllou/'
else:
    datasetpath_data1 = '/leonardo_scratch/fast/IscrB_IC-DIFF/ntriantafyllou/sub_databases/personal_projects/ICs_project/v1/'
if data2 in ['Tb_RSDs_noise10_000', 'Tb_RSDs_noise10_000_wedge1', 'Tb_RSDs_noiseAA410_000', 'Tb_RSDs_noiseAA410_000_wedge1', 'Tb_RSDs_noise_iv_filter_1000_wedge1',
'Tb_RSDs_noise_no_filter_1000_wedge1','Tb_RSDs_noise_w_filter_1000_wedge1' ,'Tb_RSDs_noise_w_filter_10_000_wedge1', 'Tb_RSDs_noise_iv_filter_1000', 'Tb_RSDs_noise_iv_filter_10_000_wedge1', 'Tb_RSDs_noise_iv_filter_100_wedge1']:
    datasetpath_data2 = '/leonardo_scratch/fast/CNHPC_1497299_0/ntriantafyllou/'
else: 
    datasetpath_data2 = '/leonardo_scratch/fast/IscrB_IC-DIFF/ntriantafyllou/sub_databases/personal_projects/ICs_project/v1/'





idds = np.arange(12701, 14701, 1)[1000:2000]
# observe = ['G','Tb']  # 'Tb', 'galaxy_counts'

batch_size = 8
BOX_SIDE = 200  # pixels per side of the cube
do_train = int(args.do_train)  # True for training, False for inference only
num_epochs = 400 # how many if checkpoint is not loaded, or how many extra epochs to train for (on top of the loaded checkpoint epoch) if continue_training is True
base_channels = 32
learning_rate = 2e-4  # originally 2e-4
N_sample = 1000 # number for sampling steps duting sampling
continue_training = False ; previous_epoch = 'last'  # number of checkpoint if continue_training=True

data1 = str(args.data1)
data2 = str(args.data2)
condition_on_NGPE_input = int(args.condition_on_NGPE)
condition_on_NGPE = ['None', 'simple_conditioning', 'residual_whole', 'residual_increments'][condition_on_NGPE_input] ; model_number = 10

box_length = BOX_SIDE # duplicate - do be fixed

model = UNet(
    in_chan=1,       # target (noised)
    cond_chan=len(observe),     # your conditioning channels
    out_chan=1,      # predict noise for target only
    base_channels=base_channels,
    ICs_NGPE=condition_on_NGPE,
).to(device)


if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DP(model) 
unet=model.to(device)



# ██████╗  █████╗ ████████╗ █████╗ 
# ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
# ██║  ██║███████║   ██║   ███████║
# ██║  ██║██╔══██║   ██║   ██╔══██║
# ██████╔╝██║  ██║   ██║   ██║  ██║
# ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝


'''--------------------------------------------Data, DataLoaders, Splitting, Normalizing-----------------------------------------------'''


#Make a dataset class for your data: this defines what you get for each batch
class PS_Dataset(Dataset):
    def __init__(self, path, idds):
        self.datapath = path
        self.idds=idds
        
    def __len__(self):
        return len(self.idds)

    def __getitem__(self, idx):
        idd = self.idds[idx]
        label = np.load(datasetpath_ICs+f'ICs/ICs_{int(idd)}.npy',         allow_pickle=True)[:BOX_SIDE, :BOX_SIDE, :BOX_SIDE]
        cond1 = np.load(datasetpath_data1+f'{data1}/{data1}_{int(idd)}.npy', allow_pickle=True)[:BOX_SIDE, :BOX_SIDE, :BOX_SIDE]
        cond=np.array([cond1])
        label=np.array([label])  
        if data2!= 'None': 
            cond2 = np.load(datasetpath_data2+f'{data2}/{data2}_{int(idd)}.npy', allow_pickle=True)[:BOX_SIDE, :BOX_SIDE, :BOX_SIDE]
            cond = np.concatenate(([cond1], [cond2]))

        # # Normalization 
        # mu_label =  np.mean(label)
        # sigma_label = np.std(label)
        # mu_cond1 =  np.mean(cond1)
        # sigma_cond1 = np.std(cond1)
        # mu_cond2 =  np.mean(cond2)
        # sigma_cond2 = np.std(cond2)
        # label = (label - mu_label) / (sigma_label + 1e-12) 
        # cond1 = (cond1 - mu_cond1) / (sigma_cond1 + 1e-12) 
        # cond2 = (cond2 - mu_cond2) / (sigma_cond2 + 1e-12) 
        if condition_on_NGPE!='None':
            NGPE_ICs = np.load(NGPE_path+f'MAP_samples/{data1}_{data2}/MAP_samples_{data1}_{data2}_model{model_number}_{int(idd)}.npy', allow_pickle=True)[:BOX_SIDE, :BOX_SIDE, :BOX_SIDE]
            cond = np.concatenate((cond, [NGPE_ICs]))  # add NGPE as last conditioning channel
            if condition_on_NGPE=='residual_whole':
                label = label - NGPE_ICs
            elif condition_on_NGPE=='residual_increments':
                pass
        return torch.Tensor(label), torch.Tensor(cond)



print('Dataset class done')

# Split dataset
# Generate shuffled indices


N = len(idds)
# indices = np.random.permutation(N)
indices = np.arange(0,N,1)

# Compute split sizes
n_train = int(0.7 * N)
n_val   = int(0.15 * N)
n_test  = N - n_train - n_val  # remainder goes to test

# Split indices
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]


train_DS = PS_Dataset(datasetpath, torch.Tensor(idds[train_idx]))
val_DS   = PS_Dataset(datasetpath, torch.Tensor(idds[val_idx]))
test_DS  = PS_Dataset(datasetpath, torch.Tensor(idds[test_idx]))


train_dataloader = DataLoader(train_DS, batch_size=batch_size, shuffle=True, generator = torch.Generator(device='cpu'), )
# num_workers=8,  pin_memory=True,  persistent_workers=True)
valid_dataloader = DataLoader(val_DS, batch_size=batch_size, shuffle=True, generator = torch.Generator(device='cpu'),)
# num_workers=8,  pin_memory=True,  persistent_workers=True)
test_dataloader = DataLoader(test_DS, batch_size=1, shuffle=True, generator = torch.Generator(device='cpu'), )
# num_workers=8,  pin_memory=True,  persistent_workers=True)



# Set up normalization ---------------------------
for label, data in train_dataloader:
    a = label
    b = data
    break
print(a.shape, b.shape)

mu_label = a.mean()
sigma_label = a.std()

mu_cond = b[:,:,:,:,:].mean(axis=[0,2,3,4])
sigma_cond = b[:,:,:,:,:].std(axis=[0,2,3,4])

mu_cond1 = mu_cond[0]
sigma_cond1 = sigma_cond[0]

if data2 != 'None':
    mu_cond2 = mu_cond[1]
    sigma_cond2 = sigma_cond[1]

if condition_on_NGPE!='None':
    mu_cond_NGPE = mu_label
    sigma_cond_NGPE = sigma_label



# initialize parameters
beta_min = 0.1
beta_max = 20.0
eps = 1e-5

if do_train==1:

    # ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
    # ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝ 
    #    ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
    #    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
    #    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
    #    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 




    scaler = torch.amp.GradScaler('cuda')    
    ema = ExponentialMovingAverage(unet.parameters(), decay=0.9999)
    # initialize optimizer
    opt = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    # remove log file if exists
    log_path = f"/leonardo_work/IscrB_IC-DIFF/ntriantafyllou/ICs/diffusion_DIY/logs/sde/checkpoints/{data1}_{data2}/cond_on_NGPE_{condition_on_NGPE}/checkpoints/"
    path = Path(log_path)
    path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path + "train_loss.log"
    
    if continue_training==True:
        if previous_epoch == 'last':
            import re, os
            previous_epoch = max(
                (int(m.group(1)) for f in os.listdir(log_path)
                for m in [re.match(r"^sde_ddpmpp_(\d+)\.pt$", f)] if m),
                default=0
            )
        if previous_epoch >= 26:
            print("Loading checkpoint from epoch", previous_epoch)
            ckpt_path = f"{log_path}sde_ddpmpp_{previous_epoch}.pt"  # pick your checkpoint
            ckpt = torch.load(ckpt_path, map_location=device)
            unet.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            ema.load_state_dict(ckpt["ema"])
            ema.to(device)
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            print("Resuming from epoch", start_epoch)
        else:
            print("No checkpoint found, starting from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0
        if os.path.isfile(log_file):
            os.remove(log_file)





    # loop
    for epoch_idx in range(start_epoch, num_epochs):
        epoch_loss = []
        for batch_idx, (label, cond) in enumerate(train_dataloader):
            unet.train()
            opt.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16): # For numerical tractability of 200^3 boxes 

                # Add this -----------------------------------------
                x_0 = label.to(device)
                cond = cond.to(device)
                # (optional) your normalizations...
                x_0 = (x_0 - mu_label) / sigma_label
                cond[:,0] = (cond[:,0] - mu_cond1) / sigma_cond1
                if data2 != 'None': cond[:,1] = (cond[:,1] - mu_cond2) / sigma_cond2
                if condition_on_NGPE!='None': cond[:,-1] = (cond[:,-1] - mu_cond_NGPE) / sigma_cond_NGPE
                # -------------------------------------------------

                # generate time t (see above for epsilon)
                b = x_0.size(dim=0)
            
                t = torch.rand(b).to(device) * (1.0 - eps) + eps  # shape (batch_size)
                # get mean and standard deviation in \tilde{x}
                mean_x = torch.exp(-0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min)  # shape (batch_size)
                std_x = torch.sqrt(1.0 - torch.exp(-0.5 * t**2 * (beta_max - beta_min) - t * beta_min))  # shape (batch_size)

                # get \tilde{x}
                z = torch.randn_like(x_0).to(device)
                x = mean_x[:, None, None, None, None] * x_0 + std_x[:, None, None, None, None] * z

                # get model output (score function's output)
                model_out = unet(x, t, cond)

                # get loss
                loss = torch.square(std_x[:, None, None, None, None] * model_out + z)
    #             loss = torch.mean(loss, dim=(1,2,3,4))
                loss = torch.mean(loss)

            scaler.scale(loss).backward()
            # Clip gradients
            scaler.unscale_(opt)
            # torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update()
            
            # log
            epoch_loss.append(loss.item())
            # print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, loss), end="\r", flush=True)


        # validation step (optional, can be done less frequently than every training step)
        for batch_idx, (label, cond) in enumerate(valid_dataloader):
            unet.eval()
            # opt.zero_grad()

            # with torch.amp.autocast('cuda', dtype=torch.float16): # For numerical tractability of 200^3 boxes 
            with torch.no_grad():
                # Add this -----------------------------------------
                x_0 = label.to(device)
                cond = cond.to(device)
                # (optional) your normalizations...
                x_0 = (x_0 - mu_label) / sigma_label
                cond[:,0] = (cond[:,0] - mu_cond1) / sigma_cond1
                if data2 != 'None': cond[:,1] = (cond[:,1] - mu_cond2) / sigma_cond2
                if condition_on_NGPE!='None': cond[:,-1] = (cond[:,-1] - mu_cond_NGPE) / sigma_cond_NGPE
                # -------------------------------------------------

                # generate time t (see above for epsilon)
                b = x_0.size(dim=0)
            
                t = torch.rand(b).to(device) * (1.0 - eps) + eps  # shape (batch_size)
                # get mean and standard deviation in \tilde{x}
                mean_x = torch.exp(-0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min)  # shape (batch_size)
                std_x = torch.sqrt(1.0 - torch.exp(-0.5 * t**2 * (beta_max - beta_min) - t * beta_min))  # shape (batch_size)

                # get \tilde{x}
                z = torch.randn_like(x_0).to(device)
                x = mean_x[:, None, None, None, None] * x_0 + std_x[:, None, None, None, None] * z

                # get model output (score function's output)
                model_out = unet(x, t, cond)

                # get loss
                val_loss = torch.square(std_x[:, None, None, None, None] * model_out + z)
    #             loss = torch.mean(loss, dim=(1,2,3,4))
                val_loss = torch.mean(val_loss)
            
            
        print("epoch{} (iter{}) - train loss {:5.4f} - val loss {:5.4f}".format(epoch_idx+1, batch_idx+1, epoch_loss[-1], val_loss), "Date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        # if epoch_idx%10 == 0:
        # plot if epoch is even
        if epoch_idx % 2 == 0:
            x_0_fake = torch.randn_like(x_0).to(device) 
            x = diff_tools.run_inference(unet, x_0_fake, cond, beta_min, beta_max, eps=0.001, N=N_sample, device=device, ICs_NGPE=condition_on_NGPE, ema=ema)
            
            if condition_on_NGPE=='residual_whole':
                x = x + cond[:, -1, :, :, :]  # add back NGPE to both true and inferred
                x_0 = x_0 + cond[:, -1, :, :, :]
            fig, k08 = diff_tools.plot_test_all(box_length, x_0, x, cond, return08=True) 

            path_png = f"/leonardo_work/IscrB_IC-DIFF/ntriantafyllou/ICs/diffusion_DIY/logs/sde/checkpoints/{data1}_{data2}/cond_on_NGPE_{condition_on_NGPE}/test_pngs/"

            path = Path(path_png)
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{path_png}test_epoch_{epoch_idx}.png", dpi=200)
            plt.close(fig)



        checkpoint_path = f"/leonardo_work/IscrB_IC-DIFF/ntriantafyllou/ICs/diffusion_DIY/logs/sde/checkpoints/{data1}_{data2}/cond_on_NGPE_{condition_on_NGPE}/checkpoints/"
        path = Path(checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # finalize epoch (save log and checkpoint)
        epoch_average_loss = sum(epoch_loss)/len(epoch_loss)
        print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx, batch_idx+1, epoch_average_loss), "Date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        # print("Date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush = True)
        with open(log_file, "a") as f:
            for l in epoch_loss:
                f.write("%s\n" %l)
    #     torch.save(unet.state_dict(), f"logs/sde/sde_ddpmpp_{epoch_idx}.pt")
        torch.save({
        "model": unet.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch_idx,
        }, f"{checkpoint_path}sde_ddpmpp_{epoch_idx}.pt")

        # # save the best model based on k08
        # if epoch_idx==start_epoch:
        #     best_k08 = k08
        #     torch.save({
        #     "model": unet.state_dict(),
        #     "ema": ema.state_dict(),
        #     "optimizer": opt.state_dict(),
        #     "scaler": scaler.state_dict(),
        #     "epoch": epoch_idx,
        #     }, f"{checkpoint_path}sde_ddpmpp_best.pt")
        # else:
        #     if k08 > best_k08:
        #         best_k08 = k08
        #         torch.save({
        #         "model": unet.state_dict(),
        #         "ema": ema.state_dict(),
        #         "optimizer": opt.state_dict(),
        #         "scaler": scaler.state_dict(),
        #         "epoch": epoch_idx,
        #         }, f"{checkpoint_path}sde_ddpmpp_best.pt")



    print("Done")









# # ██████╗ ██╗      ██████╗ ████████╗████████╗██╗███╗   ██╗ ██████╗ 
# # ██╔══██╗██║     ██╔═══██╗╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝ 
# # ██████╔╝██║     ██║   ██║   ██║      ██║   ██║██╔██╗ ██║██║  ███╗
# # ██╔═══╝ ██║     ██║   ██║   ██║      ██║   ██║██║╚██╗██║██║   ██║
# # ██║     ███████╗╚██████╔╝   ██║      ██║   ██║██║ ╚████║╚██████╔╝
# # ╚═╝     ╚══════╝ ╚═════╝    ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ 






# initialize x_0
for label, data in test_dataloader:
    x_0 = label.to(device)
    cond = data.to(device) 
    x_0 = (x_0-mu_label ) /sigma_label     # target (clean)
    cond[:,0,:,:,:] = (cond[:,0,:,:,:]-mu_cond1)/sigma_cond1
    if data2 != 'None': cond[:,1,:,:,:] = (cond[:,1,:,:,:]-mu_cond2)/sigma_cond2
    if condition_on_NGPE!='None': cond[:,-1,:,:,:] = (cond[:,-1,:,:,:]-mu_cond_NGPE)/sigma_cond_NGPE
    break
# generate images

checkpoint = torch.load(f"{checkpoint_path}sde_ddpmpp_95.pt", map_location=device)
unet.load_state_dict(checkpoint["model"])
ema = ExponentialMovingAverage(unet.parameters(), decay=0.9999)
ema.load_state_dict(checkpoint["ema"])
ema.copy_to(unet.parameters())   # <-- apply EMA weights to UNet
unet.eval()
x_0_fake = torch.randn_like(x_0).to(device) 


x = diff_tools.run_inference(unet, x_0_fake, cond, beta_min, beta_max, eps=1e-3, N=1000, device=device, ICs_NGPE=condition_on_NGPE)




x_inferred = x.detach().cpu().numpy()[0, 0]  # shape (D,H,W)
cond_np = cond.detach().cpu().numpy()[0]     # shape (2,D,H,W)


# ----------------------------------------------------
# Plotting (slices through the 3D field)
# ----------------------------------------------------
slice_idx = box_length // 2


extent = [0, box_length*1.5, 0 ,  box_length*1.5]
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(cond_np[0, slice_idx], cmap='viridis', extent=extent)
axs[0].set_title("21cm")
if cond_np.shape[0]>1:
    axs[1].imshow(cond_np[1, slice_idx], cmap='viridis', extent=extent)
    axs[1].set_title("Galaxies")
axs[2].imshow(x_inferred[slice_idx], cmap='viridis', extent=extent, vmin=-4, vmax=4)
axs[2].set_title("Inferred ICs")
axs[3].imshow(x_0[0,0, slice_idx].to('cpu'), cmap='viridis', extent=extent, vmin=-4, vmax=4)
axs[3].set_title("True ICs")
for ax in axs:
    ax.set_xlabel('x axis [Mpc]')
axs[0].set_ylabel('y axis [Mpc]')

plt.tight_layout()
# axs[3].colorbar()

plt.savefig('/leonardo_work/IscrB_IC-DIFF/ntriantafyllou/ICs/diffusion_DIY/inference_example.png', dpi=300)

plt.close()


# PDF
bins=np.linspace(-5,5,100)
x_inferred = x.detach().cpu().numpy()[0, 0]  # shape (D,H,W)
plt.hist(x_inferred.flatten(), bins=bins, alpha=0.3, label='inferred');
plt.hist(x_0.to('cpu').flatten(), bins=bins, alpha=0.3, label='true');
plt.legend()
plt.savefig('/leonardo_work/IscrB_IC-DIFF/ntriantafyllou/ICs/diffusion_DIY/pdf_example.png', dpi=300)
plt.close()

# Power Spectrum
pk1, k1 = pbox.get_power(x_inferred,box_length*1.5, log_bins=True)
pk2, k2 = pbox.get_power(x_0[0,0,:,:,:].to('cpu'), box_length*1.5, log_bins=True)

plt.loglog(k1,pk1, label='inferred')
plt.loglog(k2,pk2, label='true')
plt.legend()
plt.savefig('/leonardo_work/IscrB_IC-DIFF/ntriantafyllou/ICs/diffusion_DIY/ps_example.png', dpi=300)
plt.close()

pk, k = pbox.get_power(deltax= x_inferred, boxlength = box_length*1.5, deltax2 = x_0[0,0,:,:,:].to('cpu'), log_bins=True)
plt.plot(k,pk/np.sqrt(pk1*pk2))
plt.xscale('log')
plt.savefig('/leonardo_work/IscrB_IC-DIFF/ntriantafyllou/ICs/diffusion_DIY/ccc_example.png', dpi=300)
plt.close()



# # initialize optimizer
# opt = torch.optim.Adam(unet.parameters(), lr=2e-4, eps=1e-08)
# scheduler = torch.optim.lr_scheduler.LinearLR(
#     opt,
#     start_factor=1.0/5000,
#     end_factor=1.0,
#     total_iters=5000)

# # 1. Initialize T and alpha
# #   (See above note for precision.)
# T = 200
# alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
# alpha_bars = torch.cumprod(alphas, dim=0)
# sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
# sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)

# # remove log file if exists
# log_file = "train_loss.log"
# if os.path.exists(log_file):
#     os.remove(log_file)

# # loop
# num_epochs = 500
# for epoch_idx in range(num_epochs):
#     epoch_loss = []
#     for batch_idx, (label, cond) in enumerate(train_dataloader):
#         unet.train()
#         opt.zero_grad()

#         # 2. Pick up x_0 (shape: [batch_size, 3, 32, 32, 32])
#         x_0 = label.to(device)      # target (clean)
#         cond = cond.to(device)     # conditioning input(s)
# #         3. Pick up random timestep, t .
#         b = x_0.size(0)
#         t = torch.randint(T, (b,), device=device)
# #         4. Generate the seed of noise, epsilon .
#         eps = torch.randn_like(x_0)
# #     5. Compute x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon
#         x_t = sqrt_alpha_bars_t[t][:, None, None, None, None].float() * x_0 + \
#               sqrt_one_minus_alpha_bars_t[t][:, None, None, None, None].float() * eps

#         # 6. Get loss and apply gradient (update)
#         model_out = unet(x_t, t, cond)
#         loss = F.mse_loss(model_out, eps, reduction="mean")
#         loss.backward()
#         opt.step()
#         scheduler.step()

#         # log
#         epoch_loss.append(loss.item())
#         print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, loss), end="\r")

#     # finalize epoch (save log and checkpoint)
#     epoch_average_loss = sum(epoch_loss)/len(epoch_loss)
#     print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx+1, batch_idx+1, epoch_average_loss))
#     with open(log_file, "a") as f:
#         for l in epoch_loss:
#             f.write("%s\n" %l)
#     torch.save(unet.state_dict(), f"_checkpoints/ddpm_unet_{epoch_idx}.pt")

# print("Done")
#Make a dataset class for your data: this defines what you get for each batch
# class PS_Dataset(Dataset):
#     def __init__(self, path, idds):
#         self.datapath = path
#         self.idds=idds
        
#     def __len__(self):
#         return len(self.idds)

#     def __getitem__(self, idx):
#         idd = self.idds[idx]
#         label = np.load(datasetpath+f'ICs/ICs_{int(idd)}.npy', allow_pickle=True)[:BOX_SIDE, :BOX_SIDE, :BOX_SIDE]
#         cond1 = np.load(datasetpath+f'{data1}/{data1}_{int(idd)}.npy', allow_pickle=True)[:BOX_SIDE, :BOX_SIDE, :BOX_SIDE]
#         cond2 = np.load(datasetpath+f'{data2}/{data2}_{int(idd)}.npy', 
#                         allow_pickle=True)[:BOX_SIDE, :BOX_SIDE, :BOX_SIDE]
#         label=np.array([label])
#         cond = np.concatenate(([cond1], [cond2]))
#         cond1=np.array([cond1])        
#         cond2 = np.array([cond2])

#         if 'Tb' in observe and 'G' in observe:
#             return torch.Tensor(label), torch.Tensor(cond)  # shape of (channels,200,200,200) , 1st can be 2
#         elif 'Tb' in observe:
#             return torch.Tensor(label), torch.Tensor(cond1) # shape of (channels,200,200,200) , 1st can be 2
#         elif 'G' in observe:
#             return torch.Tensor(label), torch.Tensor(cond2) # shape of (channels,200,200,200) , 1st can be 2
        
#         # return torch.Tensor(label), torch.Tensor(cond2) # shape of (channels,200,200,200) , 1st can be 2
