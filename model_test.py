import tf_keras as tfk
import tensorflow_hub as hub
import tensorflow as tf

num_classes = 14



print(tf.__version__)

m = tfk.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-feature-vector/2",
                   trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

m.summary()