"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import torchaudio

from specaugment import spec_augment

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000
MEL_FILTERS = 128

target_dict = dict()

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target

def get_spectrogram_feature(filepath):
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()  # [length]

    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5)  # (N_FFT / 2 + 1 * T)
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0, 1)
    feat -= feat.mean()

    # T * (N_FFT / 2 + 1)
    return feat


def get_log_melspectrogram_feature(filepath, melspectrogram, amplitude_to_db):
    sig, rate = torchaudio.load_wav(filepath)  # C * time

    S = melspectrogram(sig)  # C * n_mels * time
    S = amplitude_to_db(S)  # C * nmels * time
    S = S.detach().numpy()
    S = torch.FloatTensor(S)
    feat = S.squeeze(0).transpose(0, 1)  # time * nmels
    feat -= feat.mean()

    """
    audio, sr = librosa.load(filepath, sr=None)
    #  fmax 8000일 때 보다 5000 일때 조금 느린 느낌이 있음. 만약 병목현상 발생하면 여기 들여봐야할 듯
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, window=window, n_mels=n_mels, fmax=fmax)
    S = librosa.power_to_db(S)
    feat = torch.FloatTensor(S).transpose(0, 1)  # time * melFilter
    feat -= feat.mean()  # mean = 0 로 normalize
    """

    return feat  # time * melFilter


def get_augmented_log_melspectrogram(filepath, melspectrogram, amplitude_to_DB, time_warping_para=80,
                                     frequency_masking_para=27, time_masking_para=100,
                                     frequency_mask_num=1, time_mask_num=1):
    mel_spectrogram = get_log_melspectrogram_feature(filepath, melspectrogram, amplitude_to_DB)
    augmented_feat = spec_augment(mel_spectrogram, time_warping_para, frequency_masking_para, time_masking_para,
                                  frequency_mask_num, time_mask_num)
    return augmented_feat  # time * mel_filter


def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]  # window 에서는 / -> \\
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308,
                 n_fft=512, hop_length=128, window=torch.hamming_window, n_mels=MEL_FILTERS, fmax=5000, train=False):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=n_fft,
                                                                   hop_length=hop_length, window_fn=window,
                                                                   n_mels=n_mels, f_max=fmax)
        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        if train:
            self.get_feature = get_augmented_log_melspectrogram
        else:
            self.get_feature = get_log_melspectrogram_feature

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = self.get_feature(self.wav_paths[idx], self.melspectrogram, self.amplitude_to_DB)
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

    def __del__(self):
        del self.melspectrogram, self.amplitude_to_DB, self.get_feature

def _collate_fn(batch):
    # batch = [(feat, script) * batch_size]
    # feat = [time x mel_filter]
    # target = [char_index]
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        rand_index = np.random.permutation(self.dataset_count)
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(rand_index[self.index]))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break


            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()

