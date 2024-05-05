import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio

LICK_FILE = os.path.join('samplet', 'lick_samplet', '1_lick.wav')
NOT_LICK_FILE = os.path.join('samplet', 'noLick_samplet', '1_noLick.wav')

def load_wav_48k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)
    wav = tf.squeeze(wav, axis = -1)
    sample_rate = tf.cast(sample_rate, dtype = tf.int64)
    wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out = 48000)
    return wav

wave = load_wav_48k_mono(LICK_FILE)
nwave = load_wav_48k_mono(NOT_LICK_FILE)

plt.plot(wave)
plt.plot(nwave)
plt.show()

POS = os.path.join('samplet', 'lick_samplet')
NEG = os.path.join('samplet', 'noLick_samplet')

pos = tf.data.Dataset.list_files(POS+'\*.wav')
neg = tf.data.Dataset.list_files(NEG+'\*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)