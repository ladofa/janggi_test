import tensorflow as tf
from tensorflow import keras
from functools import partial



conv2d = partial(keras.layers.Conv2D, padding='same', use_bias=False)

def residual_block(input, filters):
    z = conv2d(filters, 3)(input)
    z = keras.layers.BatchNormalization()(z)
    z = keras.activations.elu(z)
    z = conv2d(filters, 3)(z)
    z = keras.layers.BatchNormalization()(z)
    z = z + input
    z = keras.activations.elu(z)
    return z


def gen_network(filters, n_blocks):
    input = keras.layers.Input(shape=[10, 9, 16])
    z = conv2d(filters, 3)(input)
    z = keras.activations.elu(z)

    #residual tower
    for _ in range(n_blocks):
        z = residual_block(z, filters)
    
    #policy
    p = conv2d(4, 1)(z)
    p = keras.layers.BatchNormalization()(p)
    p = keras.activations.elu(p)
    p = keras.layers.Flatten()(p)
    p = keras.layers.Dense(90 * 90 + 1)(p)
    p = keras.activations.softmax(p)

    #value
    v = conv2d(1, 1)(z)
    v = keras.layers.BatchNormalization()(v)
    v = keras.activations.elu(v)
    v = keras.layers.Flatten()(v)
    v = keras.layers.Dense(256)(v)
    v = keras.activations.elu(v)
    v = keras.layers.Dense(1)(v)
    v = keras.activations.tanh(v)
    
    p_model = keras.models.Model(inputs=[input], outputs=[p])
    v_model = keras.models.Model(inputs=[input], outputs=[v])
    full_model = keras.models.Model(inputs=[input], outputs=[p, v])

    return p_model, v_model, full_model


