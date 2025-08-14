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

def load_rnd_chunk(signal):
    wlen = 3200 # window length
    sig_batch=np.zeros([wlen])
    snt_len=signal.shape[0]
    snt_beg=np.random.randint(snt_len-wlen-1) # selects random starting point in the audio
    snt_end=snt_beg+wlen # according to window length the end is defined
    sig_batch[:]=signal[snt_beg:snt_end]
    return np.array(sig_batch) # (3200)

# initialize filterbank model
checkpoint_adaptive = torch.load('./checkpoints/checkpoint_epoch_600.pth')
checkpoint_static = torch.load('./checkpoints/checkpoint_static_epoch_600.pth')
filterbank = Filterbank(N_filt=80, filt_dim=251, fs=16000, filterbank_adaptive=True)
filterbank_static = Filterbank(N_filt=80, filt_dim=251, fs=16000, filterbank_adaptive=False)
filterbank.load_state_dict(checkpoint_adaptive['filterbank_state_dict'])
filterbank_static.load_state_dict(checkpoint_static['filterbank_state_dict'])
filterbank.eval()

# init reconstruction model
conv_vae_adaptive = ConvAutoencoder()
conv_vae_adaptive.load_state_dict(checkpoint_adaptive['conv_vae_state_dict'])
conv_vae_adaptive.eval()
conv_vae_static = ConvAutoencoder()
conv_vae_static.load_state_dict(checkpoint_static['conv_vae_state_dict'])
conv_vae_static.eval()

criterion = torch.nn.L1Loss()

SAMPLE_NUMBER = 1000

loss_static_list = []
loss_adaptive_list = []

for i in range(0, SAMPLE_NUMBER):
    print(i)

    # number of audio files
    input_audio_number = np.random.randint(len(ReadList("data_lists/TIMIT_train.scp")))

    # define input audio
    wav_lst_tr=ReadList("data_lists/TIMIT_train.scp")
    input_audio_path = os.path.join("./OUTPUT_Folder", wav_lst_tr[input_audio_number])
    input_audio, fs = sf.read(input_audio_path)

    input_audio_rnd = load_rnd_chunk(input_audio)

    # # plot filters in time domain
    # filters = filterbank.get_filters()
    # plt.figure(figsize=(12, 8))
    # for i in range(min(10, filters.shape[0])):  # Plot first 10 filters
    #     plt.plot(filters[i], label=f'Filter {i}')
    # plt.title('Learned Filterbank after training') # Initialized Filterbank
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # plot filters in frequency domain
    # fs = filterbank.fs  # sampling rate
    # fft_len = 512  
    # freqs = np.linspace(0, fs/2, fft_len//2) # Frequency axis in Hz
    # plt.figure(figsize=(16, 6))
    # for f in filters:
    #     # Compute FFT
    #     F = np.fft.fft(f, n=fft_len)
    #     mag = np.abs(F[:fft_len//2])  # keep positive frequencies only
    #     # Normalize magnitude
    #     mag /= mag.max()
    #     plt.plot(freqs, mag)
    # plt.title("Learned SincNet Filters - Frequency Response")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude")
    # plt.grid(True)
    # plt.show()

    with torch.no_grad():  # disable gradient calculation for efficiency
        input_audio_exp = np.expand_dims(input_audio_rnd, axis=0)  # add batch dim
        input_audio_exp = np.expand_dims(input_audio_exp, axis=1)  # add channel dim -> (1, 1, 3200)

        filtered_output_static = filterbank_static(torch.tensor(input_audio_exp, dtype=torch.float)) # -> (1, 80, 2950)
        filtered_output_adaptive = filterbank(torch.tensor(input_audio_exp, dtype=torch.float)) # -> (1, 80, 2950)
        reconstructed_output_static = conv_vae_static(filtered_output_static) # -> (1, 1, 2950)
        reconstructed_output_adaptive = conv_vae_adaptive(filtered_output_adaptive) # -> (1, 1, 2950)

    reconstructed_np_static = reconstructed_output_static.detach().cpu().numpy()
    reconstructed_np_static = reconstructed_np_static.squeeze()
    reconstructed_np_adaptive = reconstructed_output_adaptive.detach().cpu().numpy()
    reconstructed_np_adaptive = reconstructed_np_adaptive.squeeze()

    loss_static = criterion(torch.tensor(input_audio_rnd), torch.tensor(reconstructed_np_static))
    loss_adaptive = criterion(torch.tensor(input_audio_rnd), torch.tensor(reconstructed_np_adaptive))

    # # plot
    # fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # # Plot original audio
    # axs[0].plot(input_audio.squeeze((0, 1)))
    # axs[0].set_title("Original Audio")
    # axs[0].set_ylabel("Amplitude")

    # # Plot reconstructed audio
    # axs[1].plot(reconstructed_np)
    # axs[1].set_title("Reconstructed Audio")
    # axs[1].set_ylabel("Amplitude")

    # plt.tight_layout()
    # plt.show()

    loss_static_list.append(loss_static.item())
    loss_adaptive_list.append(loss_adaptive.item())

# compute mean of loss values
print("loss static mean: ", np.mean(np.array(loss_static)))
print("loss_adaptive mean: ", np.mean(np.array(loss_adaptive)))