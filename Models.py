import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class Models:
    def __init__(self):
        pass

    def FCN(self, classCount=2, dropoutRate=0.2):

        #### Input Layer
        inputLayer = tf.keras.layers.Input(shape=(None, None, 3))

        #### Convolution Block 1
        block1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(inputLayer)
        block1 = tf.keras.layers.Dropout(dropoutRate)(block1)
        block1 = tf.keras.layers.BatchNormalization()(block1)
        block1 = tf.keras.layers.Activation("relu")(block1)

        #### Convolution Block 2
        block2 = tf.keras.layers.Conv2D(filters=classCount, kernel_size=1, strides=1)(block1)
        block2 = tf.keras.layers.Dropout(dropoutRate)(block2)
        block2 = tf.keras.layers.BatchNormalization()(block2)
        block2 = tf.keras.layers.GlobalMaxPool2D()(block2)

        #### Output/Prediction layer
        predictions = tf.keras.layers.Activation("softmax")(block2)

        model = Model(inputs=inputLayer, outputs=predictions)
        print(model.summary())


models = Models()
models.FCN()
