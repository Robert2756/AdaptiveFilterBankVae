import time
import torch
import scipy
import numpy as np

def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,fact_amp):
    
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen])
    
    snt_id_arr=np.random.randint(N_snt, size=batch_size) # N_snt: number of available audio files
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size) # data augmentation to make model robust to volume differences

    for i in range(batch_size):
        # select a random sentence from the list  (joint distribution)
        [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]]) # pick the new waveform
        signal=signal.astype(float)/32768 # nromalize waveform to [-1, 1]

        # accesing to a random chunk
        snt_len=signal.shape[0]
        snt_beg=np.random.randint(snt_len-wlen-1) # selects random starting point in the audio
        snt_end=snt_beg+wlen # according to window length the end is defined

        sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
    
    inp=torch.from_numpy(sig_batch).float()
    return inp

class AudioDataset(torch.utils.data.IterableDataset):
    def __init__(self, wav_list, wlen, data_folder, batch_size):
        self.wav_list = wav_list
        self.wlen = wlen
        self.data_folder = data_folder
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.wav_list)

    def __iter__(self):
        while True:  # or for a fixed number of batches
            yield create_batches_rnd(
                self.batch_size, self.data_folder, self.wav_list,
                len(self.wav_list), self.wlen, 0.2
            ) 