import tensorflow as tf
from tensorflow import keras
import keras_network
import game
from params import args
import os
import numpy as np

import time
import random

import mcts

model_name = 'dualres%d_%d' % (args.filters, args.n_blocks)



checkpoint = keras.callbacks.ModelCheckpoint('cp/' + model_name + '/cp', save_best_only=False, save_weights_only=True)
tensorboard = keras.callbacks.TensorBoard('logs/' + model_name)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    p_model, v_model, full_model = keras_network.gen_network(args.filters, args.n_blocks)
    

    latest = tf.train.latest_checkpoint('cp/')
    if latest != None:
        print('load latest model : ', latest)
        full_model.load_weights(latest)

    full_model.compile(loss=('sparse_categorical_crossentropy', 'mse'), optimizer='adam')


mcts.Node.set_model(full_model)


for i in range(100):
    mirrored_strategy.run(full_model, np.zeros((2, 10, 9, 16), dtype=np.float32))
    # full_model.predict(np.zeros((2, 10, 9, 16), dtype=np.float32))
    print(i)


while True:
    start = time.time()
    episode = mcts.Mcts(game.get_init_board(1, 1))
    episode.travel(500)
    print(time.time() - start)
    n = mcts.explore(episode)
    if n == -2:
        break
    episode.move(n)

