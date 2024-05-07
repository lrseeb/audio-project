import numpy as np
import tensorflow as tf
from keras import models
from itertools import groupby

from recording_helper import recors_audio, terminate
from tf_helper import preprocess_audiobuffer

loaded_model = models.load_model("lick_model")

def predict_mic():
    audio = recors_audio()
    spec = preprocess_audiobuffer(audio)
    yhat = loaded_model.predict(spec)
    yhat = [1 if prediction > 0.75 else 0 for prediction in yhat]
    print(yhat)

    # grouping predictions
    yhat = [key for key, group in groupby(yhat)]
    print(tf.math.reduce_sum(yhat))
    licks = tf.math.reduce_sum(yhat).numpy()
    print(licks)
    return licks


if __name__ == "__main__":
    for i in range(5):
        licks = predict_mic()
        print("lick detected", licks, "times :DD")

    terminate()
    