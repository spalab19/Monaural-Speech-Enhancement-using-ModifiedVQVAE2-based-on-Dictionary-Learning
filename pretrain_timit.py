import os
import time
import glob
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import vaex as vx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader


import torchvision.datasets as datasets
import torchvision.transforms as transforms

from load_timit import CustomDataset
from model import VQVAE2


class VAETrainer:
    def __init__(self, fftsize, framesize, num_hidden, dim_embedding, num_embedding,
                 beta, batchsize, epoch, learnrate, checkpoint=20):
        # init params
        self.fs = 16000
        self.fftsize = fftsize
        self.framesize = framesize
        self.num_hidden = num_hidden
        self.dim_embedding = dim_embedding
        self.num_embedding = num_embedding
        self.beta = beta
        self.batchsize = batchsize
        self.epoch = epoch
        self.learnrate = learnrate
        self.checkpoint = checkpoint
        self.in_dim = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Dataset configuration
        # train data
        train_imag_paths = glob.glob('data/TIMIT/train/imagdata/*.npy')
        train_mask_paths = glob.glob('data/TIMIT/train/maskdata/*.npy')
        # valid data
        valid_imag_paths = glob.glob('data/TIMIT/valid/imagdata/*.npy')
        valid_mask_paths = glob.glob('data/TIMIT/valid/maskdata/*.npy')
        # normalization train data (zero mean & unit variance)
        trnpath = 'dataset_info/trn{}.hdf5'
        if not os.path.isfile(trnpath.format('_summary')):
            n_chunk = 337
            idx = 0
            data = []
            for npy_file in train_imag_paths:
                data.append(np.load(npy_file))
                if len(data) == n_chunk:
                    x = np.vstack(data)
                    dict_x = {}
                    for i, split_x in enumerate(np.hsplit(x, x.shape[1])):
                        dict_x.update({f'arr{i}': split_x.flatten()})
                    ds = vx.from_arrays(**dict_x)
                    ds.export_hdf5(trnpath.format(idx))
                    del ds
                    idx +=1
                    data = []
            ds = vx.open(trnpath.format('*'))
            ds.export_hdf5(trnpath.format('_summary'))
        else:
            ds = vx.open(trnpath.format('_summary'))

        self.mean =  np.array([ds.mean(ds[key]) for key in ds.columns.keys()])
        self.std = np.array([ds.std(ds[key]) for key in ds.columns.keys()])

        trainset = CustomDataset(train_imag_paths, train_mask_paths,
                                 mean=self.mean, std=self.std, train=True)
        self.train_loader = DataLoader(trainset, batch_size=self.batchsize,
                                       shuffle=True, pin_memory=True,
                                       num_workers=4)
        validset = CustomDataset(valid_imag_paths, valid_mask_paths,
                                 mean=self.mean, std=self.std, train=False)
        self.valid_loader = DataLoader(validset, batch_size=self.batchsize,
                                       shuffle=False, pin_memory=True,
                                       num_workers=4)
        # model define
        self.model = VQVAE2(self.fftsize//2+1,
                            self.num_hidden,
                            self.num_embedding,
                            self.dim_embedding).to(self.device)
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learnrate)
        self.fbank = self.melfilter()

        print('information')
        print('-----------')
        print('#train sample {}'.format(len(train_imag_paths)))
        print('#valid sample {}'.format(len(valid_imag_paths)))
        print("")

    def run(self):
        print('model training.../')
        print('------------------')
        self.start = time.time()
        self.train_loss_list = []
        self.valid_loss_list = []
        for i in range(1, self.epoch + 1):
            train_loss = self.train()
            valid_loss = self.valid(i)

            print("Epoch:{:03d}  Train:{:.3f}  Valid:{:.3f}".format(i, train_loss, valid_loss))
            self.train_loss_list.append(train_loss)
            self.valid_loss_list.append(valid_loss)

            if i % self.checkpoint == 0:
                self._checkpoint(i)
        else:
            # plot loss
            plt.figure(figsize=(10, 6))
            sns.set_style('white')
            sns.set_context('poster')
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.plot(self.train_loss_list, color='blue', label='train')
            plt.plot(self.valid_loss_list, color='red', label='valid')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('loss/speech.png')
            # notify training summery
            elp = (time.time() - self.start) / 3600
            print("Elapsed time:{:.1f}h".format(elp))
            # save model
            torch.save(self.model.state_dict(), "model/speech_model.pth")

    def _checkpoint(self, epoch):
        checkpoint_path = "checkpoints_pretrain/{}epoch".format(epoch)
        # plot train loss
        plt.figure(figsize=(10, 6))
        sns.set_style('white')
        sns.set_context('poster')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.plot(self.train_loss_list, color='blue', label='train')
        plt.plot(self.valid_loss_list, color='red', label='valid')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(checkpoint_path+'/loss_{}.png'.format(epoch))
        plt.close()
        # save model
        torch.save(self.model.state_dict(),
                   checkpoint_path+"/checkpoint_{}.pth".format(epoch))

    def train(self):
        self.model.train()
        train_loss = 0
        for idx, (imag, _) in enumerate(self.train_loader):
            imag = imag.to(self.device, dtype=torch.float)
            self.optimizer.zero_grad()
            x_tilde, latent_loss = self.model.forward(imag)
            loss = (torch.mean((x_tilde - imag)**2) +
                                            self.beta * torch.mean(latent_loss))
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss /= (idx+1)

        return train_loss

    def valid(self, epoch):
        self.model.eval()
        valid_loss = 0
        with torch.no_grad():
            for idx, (imag, _) in enumerate(self.valid_loader):
                imag = imag.to(self.device, dtype=torch.float)
                x_tilde, latent_loss = self.model.forward(imag)
                loss = (torch.mean((x_tilde - imag)**2)
                                          + self.beta * torch.mean(latent_loss))
                valid_loss += loss.item()

                if (epoch % self.checkpoint==0) and (idx % 5 == 0):
                    checkpoint_path = "checkpoints_pretrain/{}epoch".format(epoch)
                    if not os.path.isdir(checkpoint_path):
                        os.mkdir(checkpoint_path)
                        os.mkdir(checkpoint_path+"/fig")
                    fig = plt.figure(figsize=(10, 6))
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(imag.cpu().numpy()[0, 0, :, :]*self.std+self.mean,
                               cmap='inferno', origin='lower', vmin=-63, vmax=33)
                    ax1.axis('off')
                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(x_tilde.cpu().numpy()[0, 0, :, :]*self.std+self.mean,
                               cmap='inferno', origin='lower', vmin=-63, vmax=33)
                    ax2.axis('off')
                    fig.savefig(checkpoint_path+"/fig/batch{}_{}.png".format(idx, epoch))
                    plt.close()

                    # plot 1byF filter
                    state = self.model.state_dict()
                    weight = state['encoder_b.block.0.weight']
                    w = weight.squeeze().cpu().numpy()
                    maxidx = w.argmax(axis=1)
                    idc = sorted(np.arange(len(w)), key=lambda i: maxidx[i])
                    w = w[idc]

                    _maxidx = self.fbank.argmax(axis=1)
                    _idx = sorted(np.arange(len(self.fbank)), key=lambda i: _maxidx[i])

                    plt.figure(figsize=(10, 6))
                    sns.set_style('white')
                    sns.set_context('poster')
                    plt.rcParams['xtick.direction'] = 'in'
                    plt.rcParams['ytick.direction'] = 'in'
                    plt.imshow(w.T, aspect='auto', cmap='viridis', origin='lower')
                    plt.plot(range(len(idc)), maxidx[idc], color='gold',
                             lw=0.8, ls='--', label='1byF')
                    plt.plot(range(len(_idx)), _maxidx[_idx], color='violet',
                             lw=0.8, label='mel')
                    plt.legend(frameon=True, framealpha=1)
                    plt.title('Epoch = {}'.format(epoch))
                    plt.savefig(checkpoint_path+'/fig/filter_{}.png'.format(epoch))

            valid_loss /= (idx+1)

        return valid_loss

    @staticmethod
    def _mel_to_hz(mels):
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    @staticmethod
    def _hz_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def _mel_freq(self, n_mels, fmin, fmax):
        min_mel = self._hz_to_mel(fmin)
        max_mel = self._hz_to_mel(fmax)
        mels = np.linspace(min_mel, max_mel, n_mels)
        return self._mel_to_hz(mels)

    def melfilter(self, n_mels=128, fmin=125.0, fmax=7500.0):
        fbank = np.zeros((n_mels, int(1+self.fftsize//2)))
        fftfreqs = np.linspace(0, float(self.fs)/2, int(1+self.fftsize//2),
                               endpoint=True)
        mel_f = self._mel_freq(n_mels+2, fmin=fmin, fmax=fmax)
        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]
            fbank[i] = np.maximum(0, np.minimum(lower, upper))

        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        fbank *= enorm[:, np.newaxis]
        return fbank

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('fftsize', type=int)
    parser.add_argument('framesize', type=int)
    parser.add_argument('num_hidden', type=int)
    parser.add_argument('dim_embedding', type=int)
    parser.add_argument('num_embedding', type=int)
    parser.add_argument('beta', type=float)
    parser.add_argument('batchsize', type=int)
    parser.add_argument('epoch', type=int)
    parser.add_argument('learnrate', type=float)
    parser.add_argument('checkpoint', type=int)
    args = parser.parse_args()

    VAETrainer(fftsize=args.fftsize,
               framesize=args.framesize,
               num_hidden=args.num_hidden,
               dim_embedding=args.dim_embedding,
               num_embedding=args.num_embedding,
               beta=args.beta,
               batchsize=args.batchsize,
               epoch=args.epoch,
               learnrate=args.learnrate,
               checkpoint=args.checkpoint).run()
