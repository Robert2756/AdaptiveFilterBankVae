import os
import time
import torch
import scipy
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from filterbank import Filterbank
from backbone import ConvAutoencoder

def ReadList(list_file):
    f=open(list_file,"r")
    lines=f.readlines()
    list_sig=[]
    for x in lines:
        list_sig.append(x.rstrip())
    f.close()
    return list_sig

# define input audio
input_audio_number = 1
wav_lst_tr=ReadList("data_lists/TIMIT_train.scp")
input_audio_path = os.path.join("./OUTPUT_Folder", wav_lst_tr[input_audio_number])
input_audio, fs = sf.read(input_audio_path)

def load_rnd_chunk(signal):
    wlen = 3200 # window length
    sig_batch=np.zeros([wlen])
    snt_len=signal.shape[0]
    snt_beg=np.random.randint(snt_len-wlen-1) # selects random starting point in the audio
    snt_end=snt_beg+wlen # according to window length the end is defined
    sig_batch[:]=signal[snt_beg:snt_end]
    return np.array(sig_batch) # (3200)

input_audio = load_rnd_chunk(input_audio)

# init filterbank model
checkpoint = torch.load('./checkpoints/checkpoint_epoch_500.pth')
filterbank = Filterbank(N_filt=80, filt_dim=251, fs=16000)
filterbank.load_state_dict(checkpoint['filterbank_state_dict'])
filterbank.eval()

# init reconstruction model
conv_vae = ConvAutoencoder()
conv_vae.load_state_dict(checkpoint['conv_vae_state_dict'])
conv_vae.eval()

with torch.no_grad():  # disable gradient calculation for efficiency
    input_audio = np.expand_dims(input_audio, axis=0)  # add batch dim
    input_audio = np.expand_dims(input_audio, axis=1)  # add channel dim -> (1, 1, 3200)

    filtered_output = filterbank(torch.tensor(input_audio, dtype=torch.float)) # -> (1, 80, 2950)
    print("Filtered output shape: ", np.array(filtered_output).shape)
    reconstructed_output = conv_vae(filtered_output) # -> (1, 1, 2950)
    print("Reconstructed output shape: ", np.array(reconstructed_output).shape)

print("Shape of reconstructed output: ", reconstructed_output.shape)




# Step 1: Detach and convert to NumPy
reconstructed_np = reconstructed_output.detach().cpu().numpy()

# Step 2: Remove batch and channel dimensions
# Assuming shape is (1, 1, signal_length)
reconstructed_np = reconstructed_np.squeeze()  # shape becomes (signal_length,)

# # Optional: Normalize audio to avoid clipping
# reconstructed_np = reconstructed_np / np.max(np.abs(reconstructed_np) + 1e-9)  # safe division

# plot
plt.figure(figsize=(10, 5))
plt.plot(input_audio.squeeze((0, 1)))
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(reconstructed_np)
plt.show()


# Step 3: Save to a WAV file
sf.write("reconstructed.wav", reconstructed_np, samplerate=16000)  # adjust sample rate as needed
sf.write("original.wav", input_audio.squeeze((0, 1)), samplerate=16000)

print("Input audio shape: ", input_audio.squeeze((0, 1)).shape)
print("Reconstructed audio: ", reconstructed_np.shape)