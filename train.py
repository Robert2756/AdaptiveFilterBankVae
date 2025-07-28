import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from filterbank import Filterbank
from backbone import ConvAutoencoder
from speech_dataset import AudioDataset

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
filterbank = Filterbank(N_filt=80, filt_dim=251, fs=16000)
conv_vae = ConvAutoencoder()

# initialize optimizers
optimizer_filterbank = optim.RMSprop(filterbank.parameters(), lr=0.001,alpha=0.95, eps=1e-8) 
optimizer_vae = optim.RMSprop(conv_vae.parameters(), lr=0.001, alpha=0.95, eps=1e-8)

# train loop
N_epochs = 2
N_batches = 800

# prepare dataset
data_folder = "./OUTPUT_Folder/"
dataset = AudioDataset(
    wav_list=wav_lst_tr,
    wlen=wlen,
    data_folder=data_folder,
    batch_size=Batch_size,
)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=None) # batch_size=None as the custom dataset already yields batches
criterion = nn.MSELoss()

for epoch in range(N_epochs):
  
    test_flag=0
    filterbank.train()
    conv_vae.train()

    loss_sum=0
    err_sum=0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_epochs}")

    for i, waveform_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        # waveform_batch -> (128, 3200)
        if i >= 100:
            break

        # Reset gradients
        optimizer_filterbank.zero_grad()
        optimizer_vae.zero_grad()

        # (128, 3200) -> (128, 1, 3200)
        x_filtered = filterbank(waveform_batch.unsqueeze(1)) # (128, 80, 3200)
        # print("x_filtered shape: ", np.array(x_filtered.detach()).shape)

        # Forward pass through VAE
        x_out = conv_vae(x_filtered)
        print("x_out shape: ", np.array(x_out.detach()).shape)
        print("x_filtered shape: ", np.array(x_filtered.detach()).shape)
        print("waveform batch shape: ", np.array(waveform_batch.unsqueeze(1).detach()).shape)

        # Compute reconstruction loss
        loss = criterion(x_out, waveform_batch.unsqueeze(1)[:, :, :2950]) # EXTEND TO 2950??!

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer_filterbank.step()
        optimizer_vae.step()

        # Accumulate loss
        loss_sum += loss.item()

    print(f"Epoch [{epoch+1}/{N_epochs}], Loss: {loss_sum:.4f}")


# Save model weights after each epoch
torch.save({
    'epoch': epoch + 1,
    'filterbank_state_dict': filterbank.state_dict(),
    'conv_vae_state_dict': conv_vae.state_dict(),
    'optimizer_filterbank_state_dict': optimizer_filterbank.state_dict(),
    'optimizer_vae_state_dict': optimizer_vae.state_dict(),
    'loss': loss_sum,
}, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))