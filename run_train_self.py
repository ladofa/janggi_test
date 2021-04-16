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

def self_generator():
    
    mcts.Mcts()


    file_path_list = []
    for info in os.walk(args.gibo_path):
        path = info[0]
        files = info[2]

        for fullname in files:
            _, last = os.path.splitext(fullname)
            if last.lower() != '.gib':
                continue
            file_path = path + '/' + fullname
            file_path_list.append(file_path)

    random.shuffle(file_path_list)
    
    for file_path in file_path_list:
        # print(file_path)
        gibos = gibo.read_gibo(file_path)
        for g in gibos:
            board = gibo.init_board_from_gibo(g)
            moves = g['moves']

            if len(moves) < 30:
                continue
            
            if '대국결과' not in g['info']:
                win = 0
            else:
                result = g['info']['대국결과'][0]
                if result == '초':
                    win = 1
                elif result == '한':
                    win = -1
                elif result == '楚':
                    win = 1
                elif result == '漢':
                    win = -1
                else:
                    win = 0

            replay = game.Replay(board, moves, win)

            for cur_board, dum, move, win in replay.iterator():
                state = game.get_state(cur_board, dum)
                p = game.get_index(move)
                yield state, (np.array([p]), np.array([win]))
                #좌우반전
                # flip_board = np.flip(cur_board, axis=1)
                flip_state = np.flip(state, axis=1)
                flip_move = game.flip_move(move)
                flip_p = game.get_index(flip_move)
                yield flip_state, (np.array([flip_p]), np.array([win]))

dataset = tf.data.Dataset.from_generator(
    generator=gibo_generator,
    output_types=(tf.float32, (tf.int32, tf.float32)),
    output_shapes=((10, 9, 16), ((1), (1)))
)

dataset = dataset.shuffle(3000).batch(args.batch_size, drop_remainder=True).prefetch(2)

checkpoint = keras.callbacks.ModelCheckpoint('cp/' + model_name, save_best_only=False, save_weights_only=True)
tensorboard = keras.callbacks.TensorBoard('logs/' + model_name)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    p_model, v_model, full_model = keras_network.gen_network(args.filters, args.n_blocks)
    full_model.compile(loss=('sparse_categorical_crossentropy', 'mse'), optimizer='adam')

    latest = tf.train.latest_checkpoint('cp/')
    if latest != None:
        full_model.load_weights(latest)

full_model.fit(dataset, epochs=10, callbacks=[checkpoint, tensorboard])


full_model.save_weights('saved/' + model_name)