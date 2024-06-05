import tensorflow as tf
import tensorflow_hub as hub
    
model_mobilenet = tf.keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2",
                  trainable=False, input_shape=(96,96,3)),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

model_mobilenet.compile(
    optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy','Recall', 'Precision', 'AUC'])

print(model_mobilenet.summary())