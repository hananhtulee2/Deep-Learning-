#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    29-Apr-2024 16:55:36

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from duancuoiky.customLayers.ScalingLayer import ScalingLayer

def create_model():
    imageinput = keras.Input(shape=(128,128,3))
    scaling = ScalingLayer()(imageinput)
    conv = layers.Conv2D(8, (3,3), padding="same", name="conv_")(scaling)
    batchnorm = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_")(conv)
    relu = layers.ReLU()(batchnorm)
    maxpool = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu)
    conv_1 = layers.Conv2D(16, (3,3), padding="same", name="conv_1_")(maxpool)
    batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(conv_1)
    relu_1 = layers.ReLU()(batchnorm_1)
    maxpool_1 = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_1)
    conv_2 = layers.Conv2D(32, (3,3), padding="same", name="conv_2_")(maxpool_1)
    batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(conv_2)
    relu_2 = layers.ReLU()(batchnorm_2)
    maxpool_2 = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_2)
    conv_3 = layers.Conv2D(64, (3,3), padding="same", name="conv_3_")(maxpool_2)
    batchnorm_3 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_3_")(conv_3)
    relu_3 = layers.ReLU()(batchnorm_3)
    maxpool_3 = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_3)
    conv_4 = layers.Conv2D(64, (3,3), padding="same", name="conv_4_")(maxpool_3)
    batchnorm_4 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_4_")(conv_4)
    relu_4 = layers.ReLU()(batchnorm_4)
    maxpool_4 = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_4)
    conv_5 = layers.Conv2D(128, (3,3), padding="same", name="conv_5_")(maxpool_4)
    batchnorm_5 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_5_")(conv_5)
    relu_5 = layers.ReLU()(batchnorm_5)
    maxpool_5 = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_5)
    conv_6 = layers.Conv2D(128, (3,3), padding="same", name="conv_6_")(maxpool_5)
    batchnorm_6 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_6_")(conv_6)
    relu_6 = layers.ReLU()(batchnorm_6)
    maxpool_6 = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_6)
    gapool = layers.GlobalAveragePooling2D(keepdims=True)(maxpool_6)
    fc = layers.Reshape((-1,), name="fc_preFlatten1")(gapool)
    fc = layers.Dense(64, name="fc_")(fc)
    relu_7 = layers.ReLU()(fc)
    dropout = layers.Dropout(0.200000)(relu_7)
    fc_1 = layers.Dense(8, name="fc_1_")(dropout)
    softmax = layers.Softmax()(fc_1)

    model = keras.Model(inputs=[imageinput], outputs=[softmax])
    return model
