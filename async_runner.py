import numpy as np
from params import args
import mcts
import game


class DummyModel:
    def predict(data):
        p = np.random.random((1, 8101))
        p = p / np.sum(p)
        v = np.random.random((1, 1)) * 2 - 1
        return p, v

class AsyncRunner:
    model = DummyModel()
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def commit(self, board):
        pass

    def wait(self):
        pass

def self_generator():
    plays = [mcts.Mcts(game.get_init_board(np.random.randint(4), np.random.randint(4)))
        for _ in range(args.mcts_parallel)]
    
    while True:
        for i in range(args.mcts_parallel):
            play = plays[i]
            play.travel_once()
            if play.travel_count >= args.travel_count:


            if play.root.finished != 0: #비기는 경우 생각해서 고쳐야 한다.
                dum = 1
                finished = play.root.finished
                if play.root.turn % 2 == 1:
                    finished = -finished
                for board, move in play.history:
                    if dum == 1: #turn % 2 == 0
                        state = game.get_state(board, dum)
                        pos = game.get_index(move)
                        v = finished
                    else:
                        state = game.get_state(-np.flip(board), dum)
                        pos = game.get_index(game.rot_move(move))
                        v = -finished
                    dum = -dum 
                    yield state, (pos, v)
                


    
