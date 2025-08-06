import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from filterbank import Filterbank
from backbone import ConvAutoencoder
from speech_dataset import AudioDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

def ReadList(list_file):
    f=open(list_file,"r")
    lines=f.readlines()
    list_sig=[]
    for x in lines:
        list_sig.append(x.rstrip())
    f.close()
    return list_sig

wav_lst_tr=ReadList("data_lists/TIMIT_train.scp")
wav_lst_te=ReadList("data_lists/TIMIT_test.scp")
print("wav_lst_tr length: ", len(wav_lst_tr))
print("wav_lst_te length: ", len(wav_lst_te))

# setting seed
torch.manual_seed(1234)
np.random.seed(1234)

# Converting context and shift in samples
fs = 16000
cw_len = 200
cw_shift = 10
wlen=int(fs*cw_len/1000.00) # window length
wshift=int(fs*cw_shift/1000.00) # time interval until next chunk

# Batch_dev
Batch_size=128

# initialize model
filterbank = Filterbank(N_filt=80, filt_dim=251, fs=16000).to(device)
conv_vae = ConvAutoencoder().to(device)

# initialize optimizers
# optimizer_filterbank = optim.RMSprop(filterbank.parameters(), lr=0.001,alpha=0.95, eps=1e-8) 
# optimizer_vae = optim.RMSprop(conv_vae.parameters(), lr=0.001, alpha=0.95, eps=1e-8)

# scheduler_filterbank = torch.optim.lr_scheduler.StepLR(optimizer_filterbank, step_size=25, gamma=0.9)
# scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=25, gamma=0.9)

all_params = list(filterbank.parameters()) + list(conv_vae.parameters())
optimizer = optim.RMSprop(all_params, lr=0.001, alpha=0.9, eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.95)

# train loop
N_epochs = 600 # 18*500 = 9000 steps

# prepare dataset
data_folder = "./OUTPUT_Folder/"
dataset = AudioDataset(
    wav_list=wav_lst_tr,
    wlen=wlen,
    data_folder=data_folder,
    batch_size=Batch_size,
)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=None) # batch_size=None as the custom dataset already yields batches
# criterion = nn.MSELoss()
criterion = torch.nn.L1Loss()

# loss values
loss_values = []

for epoch in range(N_epochs):
  
    test_flag=0
    filterbank.train()
    conv_vae.train()

    loss_sum=0
    err_sum=0

    loss_epoch = []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_epochs}")

    for i, waveform_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        # waveform_batch -> (128, 3200)
        waveform_batch = waveform_batch.to(device)

        # Reset gradients
        # optimizer_filterbank.zero_grad()
        # optimizer_vae.zero_grad()
        optimizer.zero_grad()

        # (128, 3200) -> (128, 1, 3200)
        x_filtered = filterbank(waveform_batch.unsqueeze(1)) # (128, 80, 3200)
        # print("x_filtered shape: ", np.array(x_filtered.detach()).shape)

        # with torch.no_grad():
        #     filter_idx = 0
        #     num_samples = 8

        #     plt.figure(figsize=(15, 10))
        #     for i in range(num_samples):
        #         plt.subplot(num_samples, 1, i + 1)
        #         plt.plot(x_filtered[i, filter_idx].detach().cpu().numpy())
        #         plt.title(f"Sample {i}, Filter {filter_idx}")
        #         plt.tight_layout()

        #     plt.show()

        # Forward pass through VAE
        x_out = conv_vae(x_filtered)
        # print("x_out shape: ", x_out.detach().shape)
        # print("x_filtered shape: ", x_filtered.detach().shape)
        # print("waveform batch shape: ", waveform_batch.unsqueeze(1).detach().shape)

        # Compute reconstruction loss
        loss = criterion(x_out, waveform_batch.unsqueeze(1))

        # x_out = x_out.detach().cpu()
        # x_true = waveform_batch.unsqueeze(1).detach().cpu()
        # print("Input waveform: mean = %.4f, std = %.4f" % (x_true.mean(), x_true.std()))
        # print("Reconstructed: mean = %.4f, std = %.4f" % (x_out.mean(), x_out.std()))

        # Plot comparison
        # plt.figure(figsize=(12, 4))
        # plt.plot(x_true[0, 0].numpy(), label='Original')
        # plt.plot(x_out[0, 0].numpy(), label='Reconstructed')
        # plt.legend()
        # plt.title("Original vs Reconstructed waveform")
        # plt.show()

        # Backpropagation
        loss.backward()

        # Update weights
        # optimizer_filterbank.step()
        # optimizer_vae.step()
        optimizer.step()

        loss_epoch.append(loss.item())

    # Accumulate loss
    loss_sum += loss.item()
    loss_values.append(sum(loss_epoch) / len(loss_epoch))

    # scheduler_filterbank.step()
    # scheduler_vae.step()
    scheduler.step()
    # current_lr = scheduler_filterbank.get_last_lr()[0]
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch [{epoch+1}/{N_epochs}], Loss: {loss_sum:.4f}, LR: {current_lr}")

# plot the loss
plt.figure(figsize=(10, 5))
plt.plot(loss_values)
plt.yscale('log')  # Set y-axis to log scale
plt.title('Loss (Log Scale)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Save model weights after each epoch
torch.save({
    'epoch': epoch + 1,
    'filterbank_state_dict': filterbank.cpu().state_dict(),
    'conv_vae_state_dict': conv_vae.cpu().state_dict(),
    # 'optimizer_filterbank_state_dict': optimizer_filterbank.state_dict(),
    # 'optimizer_vae_state_dict': optimizer_vae.state_dict(),
    'optiimizer': optimizer.state_dict(),
    'loss': loss_sum,
}, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

# moving models back to gpu
filterbank.to(device)
conv_vae.to(device)