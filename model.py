import keras
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          DepthwiseConv2D, Dropout, UpSampling2D)
from keras.models import Model, Input


def conv_block(inp, n, k1, k2, s1, s2, activation="relu"):
    
    x = Conv2D(n, k1, strides=s1, use_bias=False, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DepthwiseConv2D(k2, strides=s2, use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.1)(x)

    return x


def upconv_block(inp, n, k1, k2, s1, s2, activation="relu"):

    x = DepthwiseConv2D(k2, strides=s2, use_bias=False, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D()(x)
    
    x = Conv2D(n, k1, strides=s1, use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x


def create_segm_model(shape=(256, 256, 3)):

    inp = Input(shape=shape)

    x = conv_block(inp, 8, 1, 5, 1, 2)
    x = conv_block(x, 16, 1, 3, 1, 2)
    x = conv_block(x, 32, 1, 3, 1, 2)
    x = conv_block(x, 48, 1, 3, 1, 2)
    x = conv_block(x, 128, 1, 3, 1, 2)
    x = upconv_block(x, 48, 1, 3, 1, 1)
    x = upconv_block(x, 32, 1, 3, 1, 1)
    x = upconv_block(x, 16, 1, 3, 1, 1)
    x = upconv_block(x, 8, 1, 3, 1, 1)
    x = upconv_block(x, 3, 1, 5, 1, 1, activation="sigmoid")

    return Model(inp, x)

if __name__ == "__main__":
    model = create_segm_model()
    model.summary()