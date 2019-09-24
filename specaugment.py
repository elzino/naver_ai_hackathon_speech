import numpy as np
import random


def time_warp(feat, W=80):
    #  추후 구현
    return feat


# https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
def spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=20,
                 time_masking_para=50, frequency_mask_num=1, time_mask_num=1):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): mel spectrogram which you want to warp and mask.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 27 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    if len(mel_spectrogram.size()) == 3:
        # batch * time * melFilter
        v = mel_spectrogram.shape[1]
        tau = mel_spectrogram.shape[2]

        # Step 1 : Time warping
        warped_mel_spectrogram = time_warp(mel_spectrogram)

        # Step 2 : Time masking
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau - t)
            warped_mel_spectrogram[:, t0:t0 + t, :] = 0

        # Step 3 : Frequency masking
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v - f)
            warped_mel_spectrogram[:, :, f0:f0 + f] = 0

    else:
        # time * melFilter
        tau = mel_spectrogram.shape[0]
        v = mel_spectrogram.shape[1]

        # Step 1 : Time warping
        # 논문에서 time warping은 효과가 있긴 있으나 작다고 현실적인 상황 고려해 적용 안해도 된다고 함
        # warped_mel_spectrogram = time_warp(mel_spectrogram)

        # Step 2 : Time masking
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau - t)
            mel_spectrogram[t0:t0 + t, :] = 0

        # Step 3 : Frequency masking
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v - f)
            mel_spectrogram[:, f0:f0 + f] = 0

    return mel_spectrogram
