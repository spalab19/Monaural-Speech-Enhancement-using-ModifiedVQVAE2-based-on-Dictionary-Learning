import random
import wave
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from utils import TIMIT


class CustomDataset(Dataset):
    def __init__(self, imag_paths, mask_paths, mean, std, train=True):
        self.mean = mean
        self.std = std
        self.imag_paths = imag_paths
        self.mask_paths = mask_paths
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        # imag
        imag = np.load(self.imag_paths[index])
        imag = (imag - self.mean) / self.std
        imag = imag[:, :, np.newaxis]
        # mask
        mask = np.load(self.mask_paths[index])
        mask = (mask - self.mean) / self.std
        mask = mask[:, :, np.newaxis]

        return self.transforms(imag), self.transforms(mask)

    def __len__(self):
        return len(self.imag_paths)


class DataLoader:
    """Load data from speech dataset"""
    def __init__(self, fftsize, framesize):
        # init params
        self.imag_path = 'data/TIMIT/{}/imagdata/{}{:02}{:02}.npy'
        self.mask_path = 'data/TIMIT/{}/maskdata/{}{:02}{:02}.npy'
        self.figs_path = 'data/TIMIT/{}/figs/{}{:02}'

        self.fftsize = fftsize
        self.framesize = framesize
        self.fs = 16000
        # dataset params
        #### NOTE: Please insert your path for TIMIT corous
        self.timit = TIMIT("/.../TIMIT")
        self.trn_list = np.load("kaldi_recipe/trn.npy")
        self.dev_list = np.load("kaldi_recipe/dev.npy")

    def fit(self):
        print('train data loading.../')
        print('----------------------')
        self.timelen = 0
        for spk_id in self.trn_list.tolist():
            self.load(spk_id, mode='train')
            print('{} done.'.format(spk_id))

        print('\ntrain data summary')
        print('------------------')
        print('#spk {}'.format(np.shape(self.trn_list)[0]))
        print('time {:.1f}h\n'.format(self.timelen / 3600))

        print('valid data loading.../')
        print('----------------------')
        self.timelen = 0
        for spk_id in self.dev_list.tolist():
            self.load(spk_id, mode='valid')
            print('{} done.'.format(spk_id))

        print('\nvalid data summary')
        print('------------------')
        print('#spk {}'.format(np.shape(self.dev_list)[0]))
        print('time {:.1f}h\n'.format(self.timelen / 3600))

    def load(self, spk_id, mode):
        # import dataset
        data_gen = self.timit.sploads(code=spk_id)

        for idx, data in enumerate(data_gen, 1):
            # delete silence
            speech = self._filter(data).squeeze()
            source_rms = np.sqrt(np.mean(speech**2))
            noise = self._get_noise(len(speech), source_rms)
            sn = np.random.choice([0, 5, 10], 1, replace=True)
            mask, noise = self._snr(speech, noise, snr=sn)
            # timelen
            self.time = len(speech) / self.fs
            # clean speech
            _cleanspect = librosa.stft(speech, self.fftsize, self.fftsize//4,
                                       window='hann')
            cleanspect_abs = np.abs(_cleanspect)
            cleanspect_log = 20*np.log10(cleanspect_abs)

            _noisyspect = librosa.stft(mask, self.fftsize, self.fftsize//4,
                                       window='hann')
            noisyspect_abs = np.abs(_noisyspect)
            noisyspect_log = 20*np.log10(noisyspect_abs)

            # separate to frames
            shift = self.framesize//2
            n_frames = int(np.ceil((np.shape(cleanspect_abs)[1] -
                                    self.framesize + shift)) / shift)
            for frame in range(n_frames):
                start = frame * shift
                cleantemplate = cleanspect_log[:, start: start+self.framesize]
                noisytemplate = noisyspect_log[:, start: start+self.framesize]

                cleanpath = self.imag_path.format(mode, spk_id, idx, frame+1)
                noisypath = self.mask_path.format(mode, spk_id, idx, frame+1)

                # save as .npy
                np.save(cleanpath, cleantemplate)
                np.save(noisypath, noisytemplate)

            self.timelen += self.time

    @staticmethod
    def import_wav(name):
        wave_file = wave.open(name, 'r')
        data = wave_file.readframes(wave_file.getnframes())
        data = np.frombuffer(data, dtype="int16") / 32767.0
        wave_file.close()
        return data

    def _get_noise(self, length, source_rms, env_list=["SCAFE", "SPSQUARE",
                                                       "PRESTO", "PSTATION",
                                                       "TBUS", "TCAR"]):
        # noise environment for train
        #### NOTE: Please insert your path for DEMAND dataset
        noise_path = "/.../DEMAND/{}/ch01.wav"

        spec_env = random.choice(env_list)
        _noise = self.import_wav(noise_path.format(spec_env))
        rms = np.sqrt(np.mean(_noise**2))
        _noise = (_noise / rms) * source_rms
        start = random.choice(range(len(_noise) - length))
        noise = _noise[start: start+length]
        return noise

    @staticmethod
    def _snr(speech, noise, snr):
        inSNR = 20 * np.log10(np.sum(abs(speech), axis=None)
                              / np.sum(abs(noise), axis=None))
        noise = noise * 10**((inSNR - snr) / 20)
        mixture = speech + noise
        return mixture, noise

    @staticmethod
    def _filter(data):
        n_filter = 2000
        ave = np.convolve((data**2).flatten(),
                          np.ones(n_filter)/n_filter, mode='same')
        candidacy = ave[np.argsort(ave)[-int(len(ave) * 0.05)]]
        threshold = 10 * np.log10(candidacy.mean()) - 21.0
        idx = np.argwhere(10*np.log10(ave) > threshold).flatten()
        return data[min(idx):max(idx)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('fftsize', type=int)
    parser.add_argument('framesize', type=int)
    args = parser.parse_args()

    DataLoader(fftsize=args.fftsize, framesize=args.framesize).fit()
