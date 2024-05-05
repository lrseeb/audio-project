import os
from random import shuffle
import tensorflow as tf 
import tensorflow_io as tfio
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

LICK_FILE = os.path.join('samplet', 'lick_samplet', '1_lick.wav')
NOT_LICK_FILE = os.path.join('samplet', 'noLick_samplet', '1_noLick.wav')

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)
    wav = tf.squeeze(wav, axis = -1)
    sample_rate = tf.cast(sample_rate, dtype = tf.int64)
    wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out = 16000)
    return wav

wave = load_wav_16k_mono(LICK_FILE)
nwave = load_wav_16k_mono(NOT_LICK_FILE)

"""
plt.plot(wave)
plt.plot(nwave)
plt.show() """

POS = os.path.join('samplet', 'lick_samplet')
NEG = os.path.join('samplet', 'noLick_samplet')

pos = tf.data.Dataset.list_files(POS+'\*.wav')
neg = tf.data.Dataset.list_files(NEG+'\*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

lengths = []
for file in os.listdir(os.path.join('samplet', 'lick_samplet')):
    tensor_wave = load_wav_16k_mono(os.path.join('samplet', 'lick_samplet', file))
    lengths.append(len(tensor_wave))

tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)

def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:80000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype = tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length = 320, frame_step = 32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis = 2)
    return spectrogram, label

filepath, label = positives.shuffle(buffer_size = 10000).as_numpy_iterator().next()

spectrogram, label = preprocess(filepath, label)

"""
plt.figure(figsize = (30, 20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()"""

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size = 1000)
data = data.batch(16)
data = data.prefetch(8)

print(len(data))

train = data.take(9)
test = data.skip(9).take(2)

"""
samples, labels = train.as_numpy_iterator().next()

samples.shape

model = Sequential()
model.add(Conv2D(16, (3, 3), activation = 'relu', input_shape = (1491, 257, 1)))
model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile('Adam', loss = 'BinaryCrossentropy', metrics = [tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

model.summary()

hist = model.fit(train, epochs = 4, validation_data = test)

plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()
"""