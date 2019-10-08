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

from specaugment import spec_augment
from label_loader import original2index

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)


PAD = 0

#  mel_spectrogram parameters
N_FFT = 512
SAMPLE_RATE = 16000
MEL_FILTERS = 128
HOP_LENGTH = 128
WINDOW_FUNCTION = torch.hamming_window
F_MAX = 5000

#  spec_augment parameters
TIME_WARPING = None
FREQUENCY_MASKING = 20
TIME_MASKING = 70
FREQUENCY_MASKING_NUM = 2
TIME_MASKING_NUM = 2

target_dict = dict()


def load_targets(path):
    # 285 :
    # 혹시 제가 간 날 계산하셨던 분과 통화연결 가능할까요?:
    list_285 = ["42_0606_317_1_00981_06",
                "42_0609_317_0_01546_06",
                "42_0610_317_0_03857_06",
                "43_0607_317_0_05775_06",
                "44_0602_317_0_05120_06",
                "41_0518_317_0_09132_06",
                "41_0527_317_0_09230_06",
                "41_0603_317_0_09924_06",
                "42_0513_317_0_00139_06",
                "42_0606_317_1_00981_06",
                "42_0609_317_0_01546_06",
                "42_0610_317_0_03857_06",
                "43_0607_317_0_05775_06",
                "44_0602_317_0_05120_06", ]

    # 132 (, 65 )
    # 괄호안 내용 삭제
    list_132_delete = ["41_0513_794_0_07198_05",
                       "41_0517_850_1_08683_03",
                       "41_0601_801_0_08009_08",
                       "41_0606_800_0_07590_05",
                       "41_0606_802_0_07786_07",
                       "41_0606_802_1_07787_00",
                       "41_0606_802_1_07787_07",
                       "41_0608_796_0_07569_01",
                       "41_0610_795_2_10208_06",
                       "41_0610_796_0_09972_01",
                       "42_0527_802_0_02450_07",
                       "42_0528_802_0_01815_07",
                       "42_0528_869_0_01811_03",
                       "42_0602_796_0_02809_01",
                       "42_0610_869_0_03544_03",
                       "44_0530_805_0_04538_06",
                       "44_0606_869_0_04208_03",
                       "41_0513_796_0_07174_01", ]

    # 132 (, 65 )
    # (특정 상황)에 -> 비오는날에
    list_132_rain = ["41_0606_802_0_07786_09",
                     "42_0520_802_0_02549_09"]
    # 비 513 오 574 는 784 날 706 에 625
    rain_str = "513 574 784 706 625 "

    # 132 (, 65 )
    # (이벤트세트메뉴) 할인되나요? -> 이번달에 (이벤트세트메뉴) 할인되나요?
    list_132_insert = ["43_0523_852_0_05654_03"]
    # 이 610 번 469 달 631 에 625
    insert_str = "610 469 631 625 "

    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            if key in list_285:
                target = target.replace("285 ", "")
            elif key in list_132_delete:
                start_ind = target.find("132")
                end_ind = target.find("65") + 3
                target = target[:start_ind] + target[end_ind:]
            elif key in list_132_rain:
                start_ind = target.find("132")
                end_ind = target.find("65") + 3
                target = target[:start_ind] + rain_str + target[end_ind:]
            elif key in list_132_insert:
                target = insert_str + target
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


def get_log_melspectrogram_feature(filepath, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hamming',
                                   n_mels=MEL_FILTERS, fmax=F_MAX):
    audio, sr = librosa.load(filepath, sr=None)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, window=window, n_mels=n_mels,
                                       fmax=fmax)
    S = librosa.power_to_db(S)
    feat = torch.FloatTensor(S).transpose(0, 1)  # time * melFilter
    feat -= feat.mean()  # mean = 0 로 normalize
    return feat  # time * melFilter


def get_augmented_log_melspectrogram(filepath, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hamming',
                                     n_mels=MEL_FILTERS, fmax=F_MAX, time_warping_para=TIME_WARPING,
                                     frequency_masking_para=FREQUENCY_MASKING, time_masking_para=TIME_MASKING,
                                     frequency_mask_num=FREQUENCY_MASKING_NUM, time_mask_num=TIME_MASKING_NUM):

    mel_spectrogram = get_log_melspectrogram_feature(filepath, n_fft, hop_length, window, n_mels, fmax)
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
            token_num = int(tokens[i])
            index = original2index[token_num]
            if index == -1:
                continue
            result.append(index)
    result.append(eos_id)
    return result


class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308, train=False):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        if train:
            self.get_feature = get_augmented_log_melspectrogram
        else:
            self.get_feature = get_log_melspectrogram_feature

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = self.get_feature(self.wav_paths[idx])
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

    def __del__(self):
        del self.get_feature


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
        logger.info('loader %d start' % (self.thread_id))
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
        logger.info('loader %d stop' % (self.thread_id))


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

