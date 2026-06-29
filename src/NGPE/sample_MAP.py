import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", type=int, default=1)
parser.add_argument("--model_number", type=int, default=1)
parser.add_argument("--data1", type=str, default='Tb')
parser.add_argument("--data2", type=str, default='galaxy_counts')
# parser.add_argument("--data3", type=str, default=True)
args = parser.parse_args()
data1 = str(args.data1)  #'Tb'  # 'Tb' or 'galaxy_counts'
data2 = str(args.data2)  #'galaxy_counts'  # 'Tb' or '
if data2 == 'None': observe = [data1]
else: observe = [data1, data2]
print('data1:', data1, 'data2:', data2, flush = True)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from datetime import datetime
print("Starting date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush = True)

import torch
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Total memory (MB):", torch.cuda.get_device_properties(0).total_memory / 1e6)
    print("MultiProcessor count:", torch.cuda.get_device_properties(0).multi_processor_count)
    print("Compute capability:", torch.cuda.get_device_properties(0).major, ".", torch.cuda.get_device_properties(0).minor)
else:
    print("No CUDA GPU found.")
print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# Improving speed (optional)
# from torch.cuda.amp import autocast, GradScaler
# scaler = GradScaler()
from torch.nn.parallel import DataParallel as DP
import torch
print(torch.version.cuda)
print(torch.__version__)
print(torch.backends.cudnn.version())
print(torch.backends.cudnn.enabled)


# Custom functions
import _my_funcs
from _my_funcs import ic_utils
from _my_funcs.ic_utils import torch_hartley, load_from_checkpoint, freeze, unfreeze
import models
from models import UNet_new # Map2MapUNet, Simple3DConvNet, UNet3D,
import plots




print('Imports done', flush = True)








# ███████╗███████╗████████╗    ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗
# ██╔════╝██╔════╝╚══██╔══╝    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝
# ███████╗█████╗     ██║       ██████╔╝███████║██████╔╝███████║██╔████╔██║███████╗
# ╚════██║██╔══╝     ██║       ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║╚════██║
# ███████║███████╗   ██║       ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████║
# ╚══════╝╚══════╝   ╚═╝       ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
                                                                                

# set paths and parameters-------------------------------------------------------------------------------------------------------------
datasetpath = '/leonardo_scratch/fast/IscrB_IC-DIFF/ntriantafyllou/sub_databases/personal_projects/ICs_project/v1/'
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

saving_path = f"_save_space/Gaussian_data_pairs/{data1}_{data2}/"


# number_of_sims = 1000
# if number_of_sims == 2000:
#     saving_path  = f"_save_space/Gaussian_data_pairs_independent/{data1}_{data2}/"
    # samples_path = f"/leonardo_scratch/large/userexternal/ntriant1/ICs/samples/Gaussian_data_pairs_independent/{data1}_{data2}/"


datasetpath = '/leonardo_scratch/fast/IscrB_IC-DIFF/ntriantafyllou/sub_databases/personal_projects/ICs_project/v1/'
saving_path = f"_save_space/Gaussian_data_pairs/{data1}_{data2}/"


idds = np.arange(12701, 14701, 1)[:2000]
# idds = idds[idds != 13210]



# observe = ['Tb', 'G']  # 'Tb', 'G'
# condition_on_gaussian = bool(args.data3)  # True or False
preload_dataset = False  # set to True if you want to preload the dataset into memory (faster(?) but uses more RAM)
BOX_SIDE = 200  # 200 for final, 64 for testing
batch_size = 2
lr = 1e-2
patience = 50
factor = 0.5
threshold = 1e-2
overide_lr = False
num_epochs = 50
np.random.seed(22)

model =  models.UNet_new(box_side=BOX_SIDE, out_chan = 1 , cond_chan= len(observe) , in_chan=len(observe)) # Choose between models.Simple3DConvNet(), models.Map2MapUNet(), models.UNet3D(), models.TinyUNet3D(), models.UNet(in_chan=2, out_chan=1,  hid_chan=16, BOX_SIDE=BOX_SIDE)
model_number = int(args.model_number)  # model number for saving/loading
continue_training = True # if True, load from checkpoint and continue training
do_train = bool(args.do_train)
linear_transform_of_choice = torch_hartley # torch_hartley  # choose between torch.fft.fftn, torch_hartley, trainable_butterfly_transform


parts_to_train = [True, True] # unc and uet respectively
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
    if name=='extra_output':
        param.requires_grad = parts_to_train[0]
    else:
        param.requires_grad = parts_to_train[1]
            
    

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DP(model) 
model=model.to(device)
model_clean = model
# -------------------------------------------------------------------------------------------------------------------------------------









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
            try: 
                cond2 = np.load(datasetpath_data2+f'{data2}/{data2}_{int(idd)}.npy', allow_pickle=True)[:BOX_SIDE, :BOX_SIDE, :BOX_SIDE]
                cond = np.concatenate(([cond1], [cond2]))
            except:
                print('CORRUPTED DATA AT IDD:', idd)




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

        return torch.Tensor(label), torch.Tensor(cond)




print('Dataset class done')

# Split dataset
# Generate shuffled indices


N = len(idds)
# indices = np.random.permutation(N)
indices = np.arange(0,N,1)

# Compute split sizes
n_train = int(1 * N)
# n_val   = int(0.15 * N)
# n_test  = N - n_train - n_val  # remainder goes to test

# Split indices
train_idx = indices[:n_train]
# val_idx = indices[n_train:n_train + n_val]
# test_idx  = indices[n_train + n_val:]


train_DS = PS_Dataset(datasetpath, torch.Tensor(idds[train_idx]))
# val_DS   = PS_Dataset(datasetpath, torch.Tensor(idds[val_idx]))
# test_DS  = PS_Dataset(datasetpath, torch.Tensor(idds[test_idx]))


test_dataloader = DataLoader(train_DS, batch_size=1, shuffle=False, generator = torch.Generator(device='cpu'), )
# valid_dataloader = DataLoader(val_DS, batch_size=batch_size, shuffle=True, generator = torch.Generator(device='cpu'), )
# test_dataloader = DataLoader(test_DS, batch_size=1, shuffle=True, generator = torch.Generator(device='cpu'), ) 



def sample_from_posterior_hartley(model, cdn, n_samples=10, device='cuda', multiple_samples=False): # using it only to sample, not being used in the inference, I only need the forward transform, not the inverse one 
    model.eval()
    model.to(device)
    cdn = cdn.to(device)
    with torch.no_grad():
        pred, diag_values = model(cdn.to(device))  # pred: (batch, channels, 200, 200, 200)
        B = pred.shape[0]
        N_modes = diag_values.shape[0] 

        sample_noise_parameter_vector = 0 + torch.randn_like(diag_values) * torch.sqrt(diag_values + 1e-6)
        sample_noise_parameter_vector = sample_noise_parameter_vector.view((1, 1, BOX_SIDE, BOX_SIDE, BOX_SIDE))

        # Inverse FFT to get sampled field
        sample = linear_transform_of_choice(sample_noise_parameter_vector)
        # addig noise to the mean prediction in real space since linear transforms are distributive over addition
        if multiple_samples == False:
            return sample+pred  # shape: (batch, channels, 200, 200, 200)
        else:
            sample_noise_parameter_matrix = np.zeros((n_samples, 1,1,BOX_SIDE,BOX_SIDE,BOX_SIDE))
            for i in range(n_samples):
                sample_noise_parameter_vector               = torch.randn_like(diag_values) * torch.sqrt(diag_values + 1e-6)
                sample_noise_parameter_matrix[i, :,:,:,:,:] = sample_noise_parameter_vector.view((1, 1, BOX_SIDE, BOX_SIDE, BOX_SIDE))
            return pred.repeat(n_samples,1) + sample_noise_parameter_matrix
    

# loss_fn = loss_fn_hartley
sample_from_posterior = sample_from_posterior_hartley


'''--------------------------------------------Start Training-------------------------------------------------------------------'''

# START TRAINING ------------------------------------------------------------------------------------------------------------------
# Set training variables if not continuing from checkpoint
start_epoch = 0
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# if use_2_optimizers ==True:
#     unet_params = [p for n,p in model.named_parameters() if 'extra_output' not in n]
#     vec_params  = [p for n,p in model.named_parameters() if 'extra_output' in n]
#     optimizer = torch.optim.Adam(unet_params, lr=lr)
#     optimizer_vec  = torch.optim.Adam(vec_params,  lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, threshold=threshold)
vlosses = []
tlosses = []
plateau = 0
best_val = float("inf")

# Overwrite training variables if continuing from checkpoint

path = Path(saving_path)
path.mkdir(parents=True, exist_ok=True)

filename = saving_path+f"model_{model_number}_checkpoint.pth"
if continue_training:
    try:
        model, optimizer, scheduler, start_epoch, vlosses, tlosses, plateau = load_from_checkpoint(filename, model_clean, model_number, device=device)
        if overide_lr:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience= patience, factor= factor, threshold= threshold)
            print('did overide lr')
    except:
        pass


print(f'starting training from epoch {start_epoch}...')


filename = saving_path+f"model_{model_number}_best_checkpoint.pth"



model, optimizer, scheduler, start_epoch, vlosses, tlosses, plateau = load_from_checkpoint(filename, model_clean, model_number, device=device)
num_epochs = len(tlosses)

print('Training done', flush = True)
# Plot losses
plot_after_epoch = 2

plt.plot(np.linspace(1,num_epochs,num_epochs)[plot_after_epoch:], tlosses[plot_after_epoch:], color = 'black', label = 'train loss', linewidth = 2)
plt.plot(np.linspace(1,num_epochs,num_epochs)[plot_after_epoch:], vlosses[plot_after_epoch:], color = 'teal', label = 'validation loss', linewidth = 2)
plt.legend()
plt.grid()

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig(saving_path+fr'losses{model_number}.png', dpi=200)
plt.close()




# Eavluate on test set and plot------------------------------------------------------------------------------------------------------------------
print('starting evaluation...')

# # Draw one sample from the posterior
# true, theta = next(iter(test_dataloader))
# true = true.to(device)
# theta = theta.to(device)

# samples = []
# for i in range(100):
#     sample = sample_from_posterior(model, theta, n_samples=100, device=device)[0,0,:,:,:].cpu().detach().numpy()
#     samples.append(sample)
# samples = np.array(samples)
# sample = samples[0]  

# pred, _   = model(theta)
# true_ics  = true[0,0,:,:,:].cpu().detach().numpy()
# pred_ics  = pred[0,0,:,:,:].cpu().detach().numpy()
# data      = theta[0,0,:,:,:].cpu().detach().numpy()
# # sample    = samples[0,0,:,:,:].cpu().detach().numpy()
# print('samples shape:', samples.shape)
# print('sample shape:', sample.shape)
# added_unc = sample-pred[0,0,:,:,:].cpu().detach().numpy()

# MAP_sample = samples.mean(axis=0)
# UNC_sample = samples.std(axis=0)

MAP_datasetpath = '/leonardo_scratch/fast/INA24_C7B14/ntriantafyllou/ICs/'
path = Path(MAP_datasetpath+f'MAP_samples/{data1}_{data2}')
path.mkdir(parents=True, exist_ok=True)


for i, (true, theta) in enumerate(test_dataloader):
    if i > N:
        break
# for idd in range(1,21):
    # true, theta = next(iter(test_dataloader))
    
    idd = i + 12701
    true = true.to(device)
    theta = theta.to(device)
    pred, _   = model(theta)
    pred_ics  = pred[0,0,:,:,:].cpu().detach().numpy()
    true_ics  = true[0,0,:,:,:].cpu().detach().numpy()
    # plt.plot(pred_ics[:,100,100], label='MAP prediction')
    # plt.imshow(true_ics[:,:,100])
    # plt.savefig(MAP_datasetpath+f'MAP_samples/Example_plot_{data1}_{data2}_model{model_number}_{int(idd)}.png', dpi=200)
    # plt.close()
    np.save(MAP_datasetpath+f'MAP_samples/{data1}_{data2}/MAP_samples_{data1}_{data2}_model{model_number}_{int(idd)}.npy', pred_ics)
    # _{int(idd)}.npy', pred[0,0,:,:,:].cpu().detach().numpy())
