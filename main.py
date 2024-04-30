import os
import tensorflow as tf
import numpy as np
import seaborn as sns
import pathlib
from IPython import display
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report


training_set, validation_set = tf.keras.utils.audio_dataset_from_directory(
    directory = './samplet/lick_samplet'
)