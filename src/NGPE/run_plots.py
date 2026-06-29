# from torch_butterfly import Butterfly

# layer = Butterfly(1024, 1024, complex=False, bias=False, init='ortho')
# x = torch.randn(8, 1024)
# y = layer(x)
# print(y.shape)  # (8, 1024)




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", type=int, default=1)
parser.add_argument("--model_number", type=int, default=1)
args = parser.parse_args()


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




# Custom functions
import _my_funcs
from _my_funcs import ic_utils
from _my_funcs.ic_utils import torch_hartley, load_from_checkpoint, freeze, unfreeze
import models
from models import Map2MapUNet, Simple3DConvNet, UNet3D
import plots
# import config 


print('Imports done', flush = True)



# ███████╗███████╗████████╗    ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗
# ██╔════╝██╔════╝╚══██╔══╝    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝
# ███████╗█████╗     ██║       ██████╔╝███████║██████╔╝███████║██╔████╔██║███████╗
# ╚════██║██╔══╝     ██║       ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║╚════██║
# ███████║███████╗   ██║       ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████║
# ╚══════╝╚══════╝   ╚═╝       ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
                                                                                

# set paths and parameters-------------------------------------------------------------------------------------------------------------
datasetpath = '/leonardo_scratch/fast/IscrB_IC-DIFF/ntriantafyllou/sub_databases/personal_projects/ICs_project/v1/'
idds = np.arange(12701, 14701, 1)[:2000]
preload_dataset = False  # set to True if you want to preload the dataset into memory (faster(?) but uses more RAM)
BOX_SIDE = 200
batch_size = 16
lr = 1e-2
patience = 3
factor = 0.5
num_epochs = 5
np.random.seed(22)
model = models.TinyUNet3D() # Choose between models.Simple3DConvNet(), models.Map2MapUNet(), models.UNet3D(), models.TinyUNet3D(), models.UNet(in_chan=2, out_chan=1,  hid_chan=16, BOX_SIDE=BOX_SIDE)
model_number = int(args.model_number)  # model number for saving/loading
continue_training = True  # if True, load from checkpoint and continue training
do_train = bool(args.do_train)
linear_transform_of_choice = torch_hartley

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DP(model) 
model=model.to(device)
model_clean = model
# -------------------------------------------------------------------------------------------------------------------------------------

# # For reproducibility
# torch.manual_seed(22)
# torch.cuda.manual_seed_all(22)
# np.random.seed(22)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False














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
        label = np.load(datasetpath+f'ICs/ICs_{int(idd)}.npy', allow_pickle=True)
        cond1 = np.load(datasetpath+f'Tb/Tb_{int(idd)}.npy', allow_pickle=True)
        cond2 = np.load(datasetpath+f'galaxy_counts/Galaxy_counts_{int(idd)}.npy', 
                        allow_pickle=True)
        label=np.array([label])
        cond = np.concatenate(([cond1], [cond2]))
        cond1=np.array([cond1])        
        cond2 = np.array([cond2])
        
# #         # Normalize
#         cond1 = ic_utils.standardize_tb(cond1)
#         cond2 = ic_utils.standardize_gc(cond2)
#         label = ic_utils.standardize_ics(label)
        
        return torch.Tensor(label), torch.Tensor(cond) # shape of (1,200,200,200) , 1st can be 2



print('Dataset class done')

# Split dataset
# Generate shuffled indices


N = len(idds)
indices = np.random.permutation(N)

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

        # Clean up GPU memory
        # del true, theta, loss
        # torch.cuda.empty_cache()
        
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


def lr_schedule(optimizer, epoch_vlosses, scheduler, plateau=0, patience=8, min_lr=1e-10):
    this_vloss = epoch_vlosses[-1]
    
    if len(epoch_vlosses) == 1:
        # First epoch, nothing to compare to
        return scheduler, plateau, 0.0
    
    prev_best_vloss = np.min(epoch_vlosses[:-1])
    percent = (prev_best_vloss - this_vloss)/prev_best_vloss * 100.
    lr = optimizer.param_groups[0]["lr"]
    if lr > min_lr:
        if prev_best_vloss - this_vloss > 0:
            plateau = 0
        if prev_best_vloss - this_vloss < 0:
            plateau += 1
        if plateau > patience and lr > 1e-10:
            scheduler.step(this_vloss)
            plateau = -1*patience
    else:
        plateau += 1

    return scheduler, plateau, percent


    
# def loss_fn(model, true, cdn=None):
#     pred, noise = model(cdn, cdn=cdn)  # or model(true, cdn=cdn) depending on structure
#     return ((pred - true) ** 2).mean()

# def wrap_to_pi(angle):
#     return (angle + np.pi) % (2 * np.pi) - np.pi




#HARTLEY------------------------------------------------------------------------------------------------------------------


def loss_fn_hartley(model, true, cdn=None):
    pred, diag_values = model(cdn)

    residual = true - pred
    residual_fft = torch_hartley(residual)
    residual_fft_flat = residual_fft.flatten(start_dim=1)
    
    inv_diag = 1.0 / (diag_values + 1e-6)
    quad = (residual_fft_flat**2 * inv_diag).mean(dim=1)  # shape (B,)
    
    logdet = torch.log(diag_values + 1e-6).mean() 
    
    loss = 0.5 * quad.mean()  + 0.5* logdet.mean()
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
            sample_noise_parameter_matrix = torch.tensor(np.zeros((n_samples, 1,1, BOX_SIDE, BOX_SIDE, BOX_SIDE))).to('cpu')
            for i in range(n_samples):
                sample_noise_parameter_vector               = torch.randn_like(diag_values) * torch.sqrt(diag_values + 1e-6)
                sample_noise_parameter_matrix[i, :,:,:,:,:] = linear_transform_of_choice( sample_noise_parameter_vector.view((1, 1, BOX_SIDE, BOX_SIDE, BOX_SIDE)) ).to('cpu')
            return pred.repeat(n_samples, 1,1,1,1,1).to('cpu') + sample_noise_parameter_matrix # shape: (n_samples, batch, channels, 200, 200, 200)
    


    
    
#HARTLEY------------------------------------------------------------------------------------------------------------------

loss_fn = loss_fn_hartley
sample_from_posterior = sample_from_posterior_hartley





# ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
# ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝ 
#    ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
#    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 


# START TRAINING ------------------------------------------------------------------------------------------------------------------
# Set training variables if not continuing from checkpoint
start_epoch = 0
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
vlosses = []
tlosses = []
plateau = 0
best_val = float("inf")

# Overwrite training variables if continuing from checkpoint
if continue_training:
    try:
        filename = f"_save_space/Models/v1/model_{model_number}_checkpoint.pth"
        model, optimizer, scheduler, start_epoch, vlosses, tlosses, plateau = load_from_checkpoint(filename, model_clean, model_number, device=device)
    except:
        pass


print(f'starting training from epoch {start_epoch}...')

if do_train:
    for epoch in range(start_epoch, start_epoch + num_epochs, 1):
        train_loss = train(model, train_dataloader, optimizer, loss_fn, epoch, device=device)
        val_loss, fe = validate(model, valid_dataloader, loss_fn, epoch, sample=None, device=device)

        vlosses.append(val_loss)
        tlosses.append(train_loss)
        scheduler, plateau, percent = lr_schedule(optimizer, vlosses, scheduler, plateau)

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
            "percent": percent,
        }
        torch.save(checkpoint, f"_save_space/Models/v1/model_{model_number}_checkpoint.pth")



# END TRAINING ------------------------------------------------------------------------------------------------------------------












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
plt.legend()
plt.grid()

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.savefig(fr'_save_space/Models/v1/losses{model_number}.png', dpi=200)
plt.close()

# Eavluate on test set and plot------------------------------------------------------------------------------------------------------------------
print('starting evaluation...')

# Draw one sample from the posterior
true, theta = next(iter(test_dataloader))
true = true.to(device)
theta = theta.to(device)

samples = []
for i in range(10):
    sample = sample_from_posterior(model, theta, n_samples=100, device=device)[0,0,:,:,:].cpu().detach().numpy() # n_samples is ignored in current implementation
    samples.append(sample)
samples = np.array(samples) # shape (N_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)

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




# # Power spectra
# plots.plot_all_power_spectra(f'_save_space/all_2point_model{model_number}.pdf', sample, pred_ics, true_ics, added_unc, BOX_SIDE=200, samples=samples)
# # plots.plot_all_power_spectra(f'_save_space/all_2point_model{model_number}_correct_MAP.pdf', sample, MAP_sample, true_ics, UNC_sample, BOX_SIDE=200) # this is the version with the correct space MAP and UNC

# # Pretty images
# plots.plot_all_slices(f'_save_space/all_slices_model{model_number}.pdf', pred_ics, true_ics, data, sample, added_unc, theta)
# plots.plot_all_slices(f'_save_space/all_slices_model{model_number}_correct_MAP.pdf', MAP_sample, true_ics, data, sample, UNC_sample, theta) # this is the version with the correct space MAP and UNC
# # PDF
# plots.plot_pdf(f'_save_space/all_pdf_model{model_number}.pdf', samples, true_ics)

# #Bispectra
# plots.plot_reduced_bispectrum_vs_theta(f'_save_space/Q_vs_theta_model{model_number}.pdf', sample_boxes = samples, k1=0.1, k2=1, BoxSize=300, true_ics=np.array([true_ics])) # sample_boxes expacts shape (N_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)
# plots.plot_reduced_bispectrum_vs_theta(f'_save_space/Q_vs_theta2_model{model_number}.pdf', sample_boxes = samples, k1=0.3, k2=0.5, BoxSize=300, true_ics=np.array([true_ics])) # sample_boxes expacts shape (N_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)
# plots.plot_reduced_bispectrum_vs_k(f'_save_space/Q_vs_k_model{model_number}.pdf', sample_boxes = samples, k1=0.1, k2=1, BoxSize=300, true_ics=np.array([true_ics])) # sample_boxes expacts shape (N_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)





# samples_test = sample_from_posterior(model, theta, n_samples=10, device=device, multiple_samples=True)[:,0,0,:,:,:].cpu().detach().numpy()


# print('the shape of the samples is:', samples_test.shape)


# # Coverage tests
# num_samples = 100
# num_sims = 100
# num_dims = int(200**3)
# num_points = 10

# chosen = np.random.choice(num_dims, size=num_points, replace=False)

# supersamples = np.zeros((num_samples, num_sims, num_points))
# supertrue_ics = np.zeros((num_sims, num_points))


# # Go through num_sims
# for i in tqdm(range(num_sims)):
#     true, theta = next(iter(test_dataloader))
#     true = true.to(device)
#     theta = theta.to(device)
#     true_ics = true[0,0,:,:,:].cpu().detach().numpy()
#     print('initial theta for check:', true_ics.flatten()[:10])
#     supertrue_ics[i,:] = true_ics.flatten()[chosen] # put here MASK

#     # supersamples[:,i,:] =  sample_from_posterior(model, theta, n_samples=num_samples, device=device, multiple_samples=True)[:,0,0,:,:,:].cpu().detach().numpy().reshape(num_samples,-1)[:, chosen]

#     samples = []
#     for j in range(num_samples):
#         sample = sample_from_posterior(model, theta, n_samples=100, device=device)[0,0,:,:,:].cpu().detach().numpy() # n_samples is ignored in current implementation
#         sample = np.array(sample).flatten()[chosen] # put here MASK

#         supersamples[j, i, :] = sample

    



# plots.plot_TARP(f'_save_space/TARP_{model_number}.pdf', supersamples, supertrue_ics, sbc=True)  # supersamples should have shape (num_samples, num_sims, num_dims) # theta has shape: (num_sims, num_dims)

print("Finished date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush = True)




















# GARBAGE

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
        
#         repeated = torch.sqrt(diag_values + 1e-6).repeat(n_samples,1)
#         print('this is randlikeshape:', torch.randn_like(repeated).shape, 'this is repeat shape:', repeated.shape)
#         sample_noise_parameter_vector = 0 + repeated * torch.sqrt(diag_values + 1e-6).repeat(n_samples,1)
        
#         sample_noise_parameter_vector = sample_noise_parameter_vector.view((n_samples, 1, 1, BOX_SIDE, BOX_SIDE, BOX_SIDE))

#         # Inverse FFT to get sampled field
#         sample = linear_transform_of_choice(sample_noise_parameter_vector)
#         # addig noise to the mean prediction in real space since linear transforms are distributive over addition
#         return sample+pred.repeat(n_samples,1)  # shape: (batch, channels, 200, 200, 200)
    
