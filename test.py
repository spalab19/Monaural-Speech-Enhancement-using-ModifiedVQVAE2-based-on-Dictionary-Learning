import os
import time
import torch
import random
import wave
import struct
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import mir_eval.separation as mir
import vaex as vx

from utils import TIMIT
from itertools import islice
from model import VQVAE2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot(data, fs, size, shift, name):
    axis = np.linspace(0, len(data) / fs, len(data))

    stft_data = librosa.stft(data, size, shift, window='blackman')
    stft_data = np.abs(stft_data)+np.spacing(1)

    sns.set_style('white')
    sns.set_context('poster')

    fig = plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[20, 1], height_ratios=[1, 2])

    ax1 = plt.Subplot(fig, gs[0, 0])
    fig.add_subplot(ax1)
    ax1.plot(axis, data)
    ax1.axis([0, max(axis), -.5, .5])
    ax1.set_xlabel("Time [sec]")
    ax1.set_ylabel("Amplitude")

    ax2 = plt.Subplot(fig, gs[1, 0])
    fig.add_subplot(ax2)
    pcm = ax2.imshow(20*np.log10(stft_data), cmap='inferno', origin='lower',
                     aspect='auto', vmin=-63, vmax=33,
                     extent=[0, len(data)/fs, 0, fs/2])
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Frequency [Hz]")

    ax3 = plt.Subplot(fig, gs[1, 1])
    fig.add_axes(ax3)
    cbar = plt.colorbar(pcm, cax=ax3)
    cbar.set_label('Power[dB]', labelpad=15, rotation=270)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def save(data, name, ch=1, nbit=16, fs=16000):
    data = np.clip(data, -1, 1)
    data = [int(x * 32767.0) for x in data]
    binwave = struct.pack('h' * len(data), *data)

    wave_write = wave.open(name, 'w')
    p = (ch, nbit // 8, fs, len(binwave), 'NONE', 'not compressed')

    wave_write.setparams(p)
    wave_write.writeframes(binwave)
    wave_write.close()

    print("save complete %s" % (name))

def _filter(data):
    n_filter = 2000
    ave = np.convolve((data**2).flatten(),
                      np.ones(n_filter)/n_filter, mode='same')
    candidacy = ave[np.argsort(ave)[-int(len(ave) * 0.05)]]
    threshold = 10 * np.log10(candidacy.mean()) - 21.0
    idx = np.argwhere(10*np.log10(ave) > threshold).flatten()
    return data[min(idx):max(idx)]

def clip(data, p=21):
    mean = np.sqrt(np.mean(np.abs(data)**2, axis=1))
    mask = (20*np.log10(np.abs(data) / mean[:, np.newaxis]) > -p).astype(int)
    maskdata = data * mask
    return maskdata

def import_wav(name):
    wave_file = wave.open(name, 'r')
    data = wave_file.readframes(wave_file.getnframes())
    data = np.frombuffer(data, dtype="int16") / 32767.0
    wave_file.close()
    return data

def _snr(speech, p, noise, pad_noise, snr):
    # for padding
    length = len(speech)
    fs = 16000
    padtime = 1
    pad_length = length + fs * padtime

    inSNR = 20 * np.log10(p / np.sum(abs(noise), axis=None))

    _speech = np.zeros(pad_length, dtype="float64")
    _speech[:length] = speech

    _noise = pad_noise * 10**((inSNR - snr) / 20)
    mixture = _speech + _noise
    return mixture, _noise[:length]

def _get_noise(length, source_rms, env_list=["STRAFFIC", "PCAFETER", "TMETRO"]):
    # noise environment for train
    #### NOTE: Please insert your path for DEMAND dataset
    noise_path = "/.../DEMAND/{}/ch01.wav"
    # for padding
    fs = 16000
    padtime = 1
    pad_length = length + fs * padtime

    spec_env = random.choice(env_list)
    _noise = import_wav(noise_path.format(spec_env))
    rms = np.sqrt(np.mean(_noise**2))
    _noise = (_noise / rms) * source_rms
    start = random.choice(range(len(_noise) - pad_length))
    pad_noise = _noise[start: start+pad_length]
    noise = pad_noise[:length]
    return noise, pad_noise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('fs', type=int)
    parser.add_argument('fftsize', type=int)
    parser.add_argument('framesize', type=int)
    parser.add_argument('num_hidden', type=int)
    parser.add_argument('num_embedding', type=int)
    parser.add_argument('dim_embedding', type=int)
    parser.add_argument('--input_snr', type=int)
    parser.add_argument('--checkpoint', type=int, default=None)

    args = parser.parse_args()

    # params
    fs = args.fs
    fftsize = args.fftsize
    framesize = args.framesize
    num_hidden = args.num_hidden
    num_embedding = args.num_embedding
    dim_embedding = args.dim_embedding
    snr = args.input_snr
    num_samples = 0

    if args.checkpoint is not None:
        checkpoint_path = "checkpoints/{}iter".format(args.checkpoint)
        if not os.path.isdir(os.path.join(checkpoint_path, "result")):
            os.mkdir(os.path.join(checkpoint_path, "result"))

        savepath = checkpoint_path+"/result/{}"
        model_path = checkpoint_path+"/checkpoint_{}.pth".format(args.checkpoint)
    else:
        savepath = "result/{}"
        model_path = "model/speech_model.pth"

    # normalization train data (zero mean & unit variance)
    ds = vx.open('data/trn_summary.hdf5')
    mean =  np.array([ds.mean(ds[key]) for key in ds.columns.keys()])
    std = np.array([ds.std(ds[key]) for key in ds.columns.keys()])

    # test data
    #### NOTE: Please insert your path for TIMIT corous
    timit = TIMIT("/.../TIMIT")
    tst_list = np.load("kaldi_recipe/tst.npy")

    # load model
    model = VQVAE2(fftsize//2+1,
                   num_hidden,
                   num_embedding,
                   dim_embedding).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("model restored.\n")

    print('model testing.../')
    print('-----------------')
    for spk_id in tst_list:
        # test speaker id
        result_path = savepath.format(spk_id)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
            os.mkdir(result_path+"/fig")
            os.mkdir(result_path+"/wav")
            os.mkdir(result_path+"/eval")

        # import TIMIT dataset for test
        data_gen = timit.spkrloads(code=spk_id)
        for idx, data in enumerate(data_gen, 1):
            print("spk_id: {}_{:02}  SNR: {:02} [dB]".format(spk_id, idx, snr))
            # delete silence
            speech = data.squeeze()
            _speech = _filter(data).squeeze()
            # normalize
            rms = np.sqrt(np.mean(_speech**2))
            p = np.sum(abs(_speech), axis=None)
            noise, pads = _get_noise(len(speech), rms)
            mask, noise = _snr(speech, p, noise, pads, snr=snr)
            # input speech spectrogram
            spect = librosa.stft(speech, fftsize, fftsize//4, window='hann')
            # spect = clip(spect, p=21)
            spect_abs = np.abs(spect)
            spect_phase = np.angle(spect)

            noisyspect = librosa.stft(mask, fftsize, fftsize//4, window='hann')
            noisyspect_abs = np.abs(noisyspect)

            shift = framesize//2
            _reconst = np.zeros(shape=(np.shape(spect_abs)[0],
                                       np.shape(spect_abs)[1]+framesize))
            n_frames = int(np.ceil((np.shape(spect_abs)[1] -
                                    framesize + shift) / shift))
            num_samples += n_frames
            noisyspect_log = 20*np.log10(noisyspect_abs)

            with torch.no_grad():
                for frame in range(n_frames):
                    start = frame * shift
                    imag = np.zeros((spect_abs.shape[0], framesize))
                    temp = (noisyspect_log[:, start: start+framesize] - mean) / std
                    imag[:, :temp.shape[1]] = temp
                    imag = imag[np.newaxis, np.newaxis, :, :]
                    imag = torch.from_numpy(imag)
                    imag = imag.to(device, dtype=torch.float)
                    x_tilde, _ = model.forward(imag)

                    decoded = x_tilde.squeeze().cpu().numpy()
                    decoded = decoded*std+mean

                    _reconst[:, start: start+framesize] = decoded
                    reconst = _reconst[:, :np.shape(spect_abs)[1]]

            reconst_abs = 10**(reconst/20)
            recon_spec = reconst_abs[:, :spect_abs.shape[1]]*np.exp(1j*spect_phase)
            waveform = librosa.istft(recon_spec, fftsize//4, window='hann',
                                     length=len(speech))
            # save results
            save(speech, ch=1, nbit=16, fs=fs,
                 name=result_path+"/wav"+"/{:02}_clean.wav".format(idx))
            save(mask[:len(speech)], ch=1, nbit=16, fs=fs,
                 name=result_path+"/wav"+"/{:02}_noisy.wav".format(idx))
            save(waveform, ch=1, nbit=16, fs=fs,
                 name=result_path+"/wav"+"/{:02}_recon.wav".format(idx))
            plot(speech, fs, fftsize, fftsize//4,
                 name=result_path+"/fig"+"/{:02}_clean.png".format(idx))
            plot(mask[:len(speech)], fs, fftsize, fftsize//4,
                 name=result_path+"/fig"+"/{:02}_noisy.png".format(idx))
            plot(waveform, fs, fftsize, fftsize//4,
                 name=result_path+"/fig"+"/{:02}_recon.png".format(idx))
            # evaluation
            evals = mir.bss_eval_sources(np.array([speech, noise]),
                                         np.array([waveform, mask[:len(speech)]-waveform]))
            print("results: {}".format(evals))
            np.save(result_path+"/eval/{}.npy".format(idx), evals)

# plot evaluation values for each speaker
index = []
n_utterance = 10
ave_eval = np.zeros((3, ))
evals = np.ndarray(shape=(3, 0))
for spk_id in tst_list:
    result_path = savepath.format(spk_id)
    for idx in range(n_utterance):
        eval_ = np.load(result_path+"/eval/{}.npy".format(idx+1))[:3, 0]
        ave_eval = ave_eval + eval_

    ave_eval = ave_eval / n_utterance
    index.append(spk_id)
    evals = np.c_[evals, ave_eval]

df = pd.DataFrame(evals.T, columns=['SDR', 'SIR', 'SAR'], index=index)
ave = df.mean()
print('summary')
print('-------')
print('#test sample {}'.format(num_samples))
print('')
print(ave)
