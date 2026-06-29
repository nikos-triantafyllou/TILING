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

number_of_sims = 1000
if number_of_sims == 2000 : 
    saving_path = f"_save_space/Gaussian_data_pairs_independent/{data1}_{data2}/" 
idds = np.arange(12701, 14701, 1)[:number_of_sims]
# observe = ['Tb', 'G']  # 'Tb', 'G'
# condition_on_gaussian = bool(args.data3)  # True or False
preload_dataset = False  # set to True if you want to preload the dataset into memory (faster(?) but uses more RAM)
BOX_SIDE = 200  # 200 for final, 64 for testing
batch_size = 4
lr = 1e-2
patience = 40
factor = 0.5
threshold = 1e-2
overide_lr = False
num_epochs = 100 # how many extra epochs to train for (on top of the loaded checkpoint epoch) if continue_training is True
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

        return torch.Tensor(label), torch.Tensor(cond)




print('Dataset class done')

# Split dataset
# Generate shuffled indices


N = len(idds)
# indices = np.random.permutation(N)
indices = np.arange(0, N, 1)

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
valid_dataloader = DataLoader(val_DS, batch_size=batch_size, shuffle=True, generator = torch.Generator(device='cpu'), )
# num_workers=8,  pin_memory=True,  persistent_workers=True)
test_dataloader = DataLoader(test_DS, batch_size=1, shuffle=True, generator = torch.Generator(device='cpu'), ) 
# num_workers=8,  pin_memory=True,  persistent_workers=True)













# ███████╗███████╗████████╗    ██╗   ██╗██████╗     ████████╗██████╗  █████╗ ██╗███╗   ██╗
# ██╔════╝██╔════╝╚══██╔══╝    ██║   ██║██╔══██╗    ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║
# ███████╗█████╗     ██║       ██║   ██║██████╔╝       ██║   ██████╔╝███████║██║██╔██╗ ██║
# ╚════██║██╔══╝     ██║       ██║   ██║██╔═══╝        ██║   ██╔══██╗██╔══██║██║██║╚██╗██║
# ███████║███████╗   ██║       ╚██████╔╝██║            ██║   ██║  ██║██║  ██║██║██║ ╚████║
# ╚══════╝╚══════╝   ╚═╝        ╚═════╝ ╚═╝            ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝




'''--------------------------------------------Setting up training process----------------------------------------------------------------'''

# Training loop
def train(model, 
          train_dataloader, 
          optimizer, 
          criterion, 
          epoch, 
          device='cpu'):
    # set model weights to be trainable
    model.train()
    tmp = []
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=False)
    for true, theta in pbar:
        # zero your gradient at each batch
        optimizer.zero_grad(set_to_none=True)
        theta = theta.to(device)
        true = true.to(device)
        # evaluate loss and backpropagate
        loss = criterion(model, true, cdn=theta)
        loss.backward()
        optimizer.step()

        # with autocast():
        #     loss = criterion(model, true, cdn=theta)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        tmp.append(loss.item())

        


        pbar.set_postfix(total_loss=loss.item())

        
    return np.mean(tmp[-500:])
 
def validate(model, valid_dataloader, criterion, epoch, sample, kperp=None,kpar=None, fe_every=10, save=None, device='cpu'):
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        model.eval()
        vtmp = []
        fe = np.nan
        pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch}")
        for i, (true, theta) in enumerate(pbar):
            theta, true = theta.to(device), true.to(device)
            vloss = criterion(model, true, cdn=theta)
            vtmp.append(vloss.item())
            pbar.set_postfix(vloss=np.mean(vtmp))
        return np.mean(vtmp), fe






#HARTLEY------------------------------------------------------------------------------------------------------------------


def loss_fn_hartley(model, true, cdn=None):
    pred, diag_values = model(cdn)

    residual = true - pred
    residual_fft = linear_transform_of_choice(residual)
    residual_fft_flat = residual_fft.flatten(start_dim=1)
    
    inv_diag = 1.0 / (diag_values + 1e-6)
    quad = (residual_fft_flat**2 * inv_diag).mean(dim=1)  # shape (B, 200**3)
    
    logdet = torch.log(diag_values + 1e-6).mean() 
    
    loss = 0.5 * quad.mean() + 0.5* logdet.mean()
    # print('quad:', quad.mean(), 'logdet:', logdet.mean())
    return loss


def loss_fn_hartley(model, true, cdn=None):
    pred, diag = model(cdn)  # (1,N) positive
    r = torch_hartley(true - pred).flatten(1)  # (1,N)

    diag = diag + 1e-6
    loss = 0.5 * (((r*r) / diag) + torch.log(diag)).mean()
    return loss


    

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
    

loss_fn = loss_fn_hartley
sample_from_posterior = sample_from_posterior_hartley

#HARTLEY------------------------------------------------------------------------------------------------------------------


print('checking hartley')
x = torch.randn(1,1,32,32,32, device='cuda')
z = torch_hartley(torch_hartley(x))
print((z - x).abs().mean().item())



# ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
# ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝ 
#    ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
#    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 


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





if do_train:
    for epoch in range(start_epoch, start_epoch + num_epochs, 1):
        train_loss = train(model, train_dataloader, optimizer, loss_fn, epoch, device=device)
        val_loss, fe = validate(model, valid_dataloader, loss_fn, epoch, sample=None, device=device)

        vlosses.append(val_loss)
        tlosses.append(train_loss)
        # scheduler, plateau, percent = lr_schedule(optimizer, vlosses, scheduler, plateau)
        scheduler.step(val_loss)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        
        # Save updated checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_losses": tlosses,
            "val_losses": vlosses,
            "ngpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "lr": lr,
            "batch_size": batch_size,
            "model_number": model_number,
            "model_name": model.__class__.__name__,
            "patience": patience,
            "factor": factor,
            "plateau": plateau,
            "device": device,
            "threshold": threshold,
            # "percent": percent,
        }
        torch.save(checkpoint, saving_path+f"model_{model_number}_checkpoint.pth")
        if vlosses[-1] == min(vlosses):
            torch.save(checkpoint, saving_path+f"model_{model_number}_best_checkpoint.pth")
            print(f"--> Saved best model checkpoint at epoch {epoch} with val loss {vlosses[-1]:.4f}")


# END TRAINING ------------------------------------------------------------------------------------------------------------------

filename = saving_path+f"model_{model_number}_best_checkpoint.pth"








# ██████╗ ██╗      ██████╗ ████████╗████████╗██╗███╗   ██╗ ██████╗ 
# ██╔══██╗██║     ██╔═══██╗╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝ 
# ██████╔╝██║     ██║   ██║   ██║      ██║   ██║██╔██╗ ██║██║  ███╗
# ██╔═══╝ ██║     ██║   ██║   ██║      ██║   ██║██║╚██╗██║██║   ██║
# ██║     ███████╗╚██████╔╝   ██║      ██║   ██║██║ ╚████║╚██████╔╝
# ╚═╝     ╚══════╝ ╚═════╝    ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                                 
'''--------------------------------------------Loss plotting and Extra plots------------------------------------------------------'''




model, optimizer, scheduler, start_epoch, vlosses, tlosses, plateau = load_from_checkpoint(filename, model_clean, model_number, device=device)
num_epochs = len(tlosses)

print('Training done', flush = True)
# Plot losses
plot_after_epoch = 2

plt.plot(np.linspace(1,num_epochs,num_epochs)[plot_after_epoch:], tlosses[plot_after_epoch:], color = 'black', label = 'train loss', linewidth = 2)
plt.plot(np.linspace(1,num_epochs,num_epochs)[plot_after_epoch:], vlosses[plot_after_epoch:], color = 'teal', label = 'validation loss', linewidth = 2)
plt.ylim(min(vlosses[plot_after_epoch:]), 1.5)
plt.legend()
plt.grid()

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig(saving_path+fr'losses{model_number}.png', dpi=200)
plt.close()




# Eavluate on test set and plot------------------------------------------------------------------------------------------------------------------
print('starting evaluation...')

# Draw one sample from the posterior
true, theta = next(iter(test_dataloader))
true = true.to(device)
theta = theta.to(device)

samples = []
for i in range(100):
    sample = sample_from_posterior(model, theta, n_samples=100, device=device)[0,0,:,:,:].cpu().detach().numpy()
    samples.append(sample)
samples = np.array(samples)
sample = samples[0]  

pred, _   = model(theta)
true_ics  = true[0,0,:,:,:].cpu().detach().numpy()
pred_ics  = pred[0,0,:,:,:].cpu().detach().numpy()
data      = theta[0,0,:,:,:].cpu().detach().numpy()
# sample    = samples[0,0,:,:,:].cpu().detach().numpy()
print('samples shape:', samples.shape)
print('sample shape:', sample.shape)
added_unc = sample-pred[0,0,:,:,:].cpu().detach().numpy()

MAP_sample = samples.mean(axis=0)
UNC_sample = samples.std(axis=0)


# # Plot 1 --------------------------------------------------------------------------------------------------------------------------------------
# plots.plot_slices(f'_save_space/posterior_example_model{model_number}.png',pred_ics, true_ics, data, sample, added_unc, theta)

# # Plot 2 --------------------------------------------------------------------------------------------------------------------------------------
# plots.plot_pdf(f'_save_space/posterior_example_model{model_number}_pdf.png', sample, pred_ics, true_ics)

# # Plot 3 --------------------------------------------------------------------------------------------------------------------------------------
# plots.plot_power_spectrum(f'_save_space/posterior_example_model{model_number}_pk.png', sample, pred_ics, true_ics, added_unc, BOX_SIDE=BOX_SIDE)

plots.plot_all_power_spectra(saving_path+f'all_2point_model{model_number}.pdf', sample, pred_ics, true_ics, added_unc, BOX_SIDE=BOX_SIDE, samples=samples)


# plots.plot_all_power_spectra(f'_save_space/all_2point_model{model_number}_correct_MAP.pdf', sample, MAP_sample, true_ics, UNC_sample, BOX_SIDE=200) # this is the version with the correct space MAP and UNC

plots.plot_all_slices(saving_path+f'all_slices_model{model_number}.pdf', pred_ics, true_ics, data, sample, added_unc, theta)
plots.plot_all_slices(saving_path+f'all_slices_model{model_number}_correct_MAP.pdf', MAP_sample, true_ics, data, sample, UNC_sample, theta) # this is the version with the correct space MAP and UNC

plots.plot_pdf(saving_path+f'all_pdf_model{model_number}.pdf', samples, true_ics)


# plots.plot_reduced_bispectrum_vs_theta(f'_save_space/Q_vs_theta_model{model_number}.pdf', sample_boxes = samples, k1=0.1, k2=1, BoxSize=300, true_ics=np.array([true_ics])) # sample_boxes expacts shape (N_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)
# plots.plot_reduced_bispectrum_vs_theta(f'_save_space/Q_vs_theta2_model{model_number}.pdf', sample_boxes = samples, k1=0.3, k2=0.5, BoxSize=300, true_ics=np.array([true_ics])) # sample_boxes expacts shape (N_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)
# plots.plot_reduced_bispectrum_vs_k(f'_save_space/Q_vs_k_model{model_number}.pdf', sample_boxes = samples, k1=0.1, k2=1, BoxSize=300, true_ics=np.array([true_ics])) # sample_boxes expacts shape (N_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)

print("Finished date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush = True)

# save all inputs to the plot function
# np.savez(saving_path+f'all_inputs_model{model_number}.npz', pred_ics=pred_ics, true_ics=true_ics, data=data, sample=sample, added_unc=added_unc, theta=theta.cpu().detach().numpy(), samples=samples)




































# GARBAGE ------------------------------------------------------------------------------------------------------------------

# def lr_schedule(optimizer, epoch_vlosses, scheduler, plateau=0, patience=8, min_lr=1e-10):
#     this_vloss = epoch_vlosses[-1]
    
#     if len(epoch_vlosses) == 1:
#         # First epoch, nothing to compare to
#         return scheduler, plateau, 0.0
    
#     prev_best_vloss = np.min(epoch_vlosses[:-1])
#     percent = (prev_best_vloss - this_vloss)/prev_best_vloss * 100.
#     lr = optimizer.param_groups[0]["lr"]
#     if lr > min_lr:
#         if prev_best_vloss - this_vloss > 0:
#             plateau = 0
#         if prev_best_vloss - this_vloss < 0:
#             plateau += 1
#         if plateau > patience and lr > 1e-10:
#             scheduler.step(this_vloss)
#             plateau = -1*patience
#     else:
#         plateau += 1

#     return scheduler, plateau, percent


    
# def loss_fn(model, true, cdn=None):
#     pred, noise = model(cdn, cdn=cdn)  # or model(true, cdn=cdn) depending on structure
#     return ((pred - true) ** 2).mean()

# def wrap_to_pi(angle):
#     return (angle + np.pi) % (2 * np.pi) - np.pi





# # For reproducibility
# torch.manual_seed(22)
# torch.cuda.manual_seed_all(22)
# np.random.seed(22)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False



# # Plot 1 --------------------------------------------------------------------------------------------------------------------------------------
# plots.plot_slices(f'_save_space/posterior_example_model{model_number}.png',pred_ics, true_ics, data, sample, added_unc, theta)

# # Plot 2 --------------------------------------------------------------------------------------------------------------------------------------
# plots.plot_pdf(f'_save_space/posterior_example_model{model_number}_pdf.png', sample, pred_ics, true_ics)

# # Plot 3 --------------------------------------------------------------------------------------------------------------------------------------
# plots.plot_power_spectrum(f'_save_space/posterior_example_model{model_number}_pk.png', sample, pred_ics, true_ics, added_unc, BOX_SIDE=BOX_SIDE)



# def sample_from_posterior_hartley(model, cdn, n_samples=10, device='cuda'): # using it only to sample, not being used in the inference, I only need the forward transform, not the inverse one 
#     model.eval()
#     model.to(device)
#     cdn = cdn.to(device)
#     with torch.no_grad():
#         pred, diag_values = model(cdn.to(device))  # pred: (batch, channels, 200, 200, 200)
#         B = pred.shape[0]
#         N_modes = diag_values.shape[0] 

#         sample_noise_parameter_vector = 0 + torch.randn_like(diag_values) * torch.sqrt(diag_values + 1e-6)
#         sample_noise_parameter_vector = sample_noise_parameter_vector.view((1, 1, BOX_SIDE, BOX_SIDE, BOX_SIDE))

#         # Inverse FFT to get sampled field
#         sample = linear_transform_of_choice(sample_noise_parameter_vector)
#         # addig noise to the mean prediction in real space since linear transforms are distributive over addition
#         return sample+pred  # shape: (batch, channels, 200, 200, 200)







# # import config 
# import importlib
# from pathlib import Path

# import torch

# __version__ = '0.0.0'

# for library in ['_version', '_butterfly']:
#     torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
#         # need str(Path) otherwise it can't find it
#         library, [str(Path(__file__).absolute().parent)]).origin)

# def check_cuda_version():
#     if torch.version.cuda is not None:  # pragma: no cover
#         cuda_version = torch.ops.torch_butterfly.cuda_version()

#         if cuda_version == -1:
#             major = minor = 0
#         elif cuda_version < 10000:
#             major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
#         else:
#             major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
#         t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

#         if t_major != major or t_minor != minor:
#             raise RuntimeError(
#                 f'Detected that PyTorch and torch_butterfly were compiled with '
#                 f'different CUDA versions. PyTorch has CUDA version '
#                 f'{t_major}.{t_minor} and torch_butterfly has CUDA version '
#                 f'{major}.{minor}. Please reinstall the torch_butterfly that '
#                 f'matches your PyTorch install.')

# check_cuda_version()
# from .butterfly import Butterfly, ButterflyUnitary, ButterflyBmm  # noqa
# from .butterfly_base4 import ButterflyBase4  # noqa
# from .multiply import butterfly_multiply  # noqa
# from . import combine
# from . import complex_utils
# from . import diagonal
# from . import permutation
# from . import special
# from . import multiply_base4

# __all__ = [
#     'Butterfly',
#     'ButterflyUnitary',
#     'ButterflyBmm',
#     'ButterflyBase4',
#     'butterfly_multiply',
#     '__version__',
# ]

# from torch_butterfly.butterfly import ButterflyUnitary


# freeze_unc = False
# freeze_unet = True
# if freeze_unc == True:
#     freeze(model.extra_output)
# if freeze_unet == True:
#     # freeze(model)
#     freeze(model.enc)
#     freeze(model.bottleneck)
#     freeze(model.up)
#     freeze(model.dec)
#     freeze(model.final)
#     # unfreeze(model.extra_output)
# for param in model.parameters():
#     param.requires_grad = False
#     print(param)

# import torch.distributed as dist #test
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# def setup():
#     dist.init_process_group(backend="nccl")  # or "gloo" if NCCL is buggy
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
# def cleanup():
#     dist.destroy_process_group()
# setup()
# device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

        # if 'Tb' in observe and 'G' in observe:
        #     return torch.Tensor(label), torch.Tensor(cond)  # shape of (channels,200,200,200) , 1st can be 2
        # elif 'Tb' in observe:
        #     return torch.Tensor(label), torch.Tensor(cond1) # shape of (channels,200,200,200) , 1st can be 2
        # elif 'G' in observe:
        #     return torch.Tensor(label), torch.Tensor(cond2) # shape of (channels,200,200,200) , 1st can be 2
        
        # # return torch.Tensor(label), torch.Tensor(cond2) # shape of (channels,200,200,200) , 1st can be 2
