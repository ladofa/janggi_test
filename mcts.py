import game
import numpy as np


class Node:
    puct = 1.0
    temp = 1.0
    max_turn = 150
    def get_pos_value(board, moves, dum):
        pos = np.ones(len(moves), dtype=np.float32) / len(moves)
        value = 0
        return pos, value
    def set_model(full_model):
        def func(board, moves, dum):
            state = game.get_state(board, dum)
            pos_row, value_row = full_model.predict(state[None, ...])
            pos = pos_row[0][[game.get_index(m) for m in moves]]
            pos = pos / np.sum(pos)
            value = value_row[0][0]
            return pos, value
        Node.get_pos_value = func

    def __init__(self, board, turn, parent=None, dum=True, prev_move=game.MOVE_EMPTY):
        self.board = board
        
        self.turn = turn
        self.parent = parent
        if parent:
            self.dum = not parent.dum
        else:
            self.dum = dum
        self.prev_move = prev_move

        self.finished = game.is_finished(board)
        if turn >= Node.max_turn:
            score = game.get_score(board) + (1.5 if turn % 2 == 0 else 0)
            if score > 0:
                self.finished = 1
            else:
                self.finished = -1
        if self.finished != 0:
            self.value = self.finished
            # self.moves = None
            # self.pos = None
            # self.children = None
            # self.total_q = None
            # self.avr_q = None
            # self.visited = None
        else:
            self.moves = game.get_all_moves(board)[0]
            #get pos and value
            self.pos, self.value = Node.get_pos_value(board, self.moves, self.dum)
            self.children = [None for _ in range(len(self.moves))]

            #자식 노드 방문횟수, 점수 등을 부모 노드에 저장
            self.total_q = np.zeros(len(self.moves), dtype=np.float32) # Q(s, a)
            self.avr_q = np.zeros(len(self.moves), dtype=np.float32) #W(s, a)
            self.visited = np.zeros(len(self.moves), dtype=np.int32) #N(s, a)
        

    def deliver(self, i):
        next_board = game.get_next_board_rot(self.board, self.moves[i])
        child = Node(next_board, self.turn + 1, self, not self.dum, self.moves[i])
        self.children[i] = child
        return child
    
    def get_next_node_index(self):
        q = self.avr_q
        u = Node.puct * self.pos * np.sqrt(np.sum(self.visited)) / (1 + self.visited)
        return np.argmax(q + u)

    def add_value(self, i, value):
        self.visited[i] += 1
        self.total_q[i] += value
        self.avr_q[i] = self.total_q[i] / self.visited[i]

    def get_choice(self):
        if Node.temp == 0:
            return np.argmax(self.visited)
        else:
            v = self.visited ** (1 / Node.temp)
            return np.random.choice(range(len(v)), p=v)


class Mcts:
    def __init__(self, board):
        self.root = Node(board, 0, parent=None, dum=True)
        self.history = []
        self.travel_count = 0

    def travel_once(self):
        node = self.root
        visited_nodes = []
        selections = []
        while True:
            if node.finished:
                leaf_value = -node.value
                break
            visited_nodes.insert(0, node)
            i = node.get_next_node_index()
            selections.insert(0, i)

            # 자식 노드 방문
            if node.children[i] != None:
                node = node.children[i]
            # leaf node - 출산
            else:
                node = node.deliver(i)
                leaf_value = -node.value
                break

        leaf_value = -visited_nodes[-1].value

        for node, i in zip(visited_nodes, selections):
            node.add_value(i, leaf_value)
            leaf_value = -leaf_value

        self.travel_count += 1
    
    def travel(self, n):
        for _ in range(n):
            self.travel_once()

    def is_finished(self):
        return self.root.finished

    def move(self, i=-1):
        node = self.root
        
        if node.finished:
            return node.finished

        if i == -1:
            i = node.get_choice()
        if node.children[i] == None:
            child = node.deliver(i)
        else:
            child = node.children[i]

        self.history.append((node.board, i))
        self.root = child
        self.travel_count = 0
        child.parent = None #garbage collecting

        return child.finished
    

def explore(mcts):
    node = mcts.root
    history = [node]
    move = []
    while True:
        if node != None:
            print(node.board)
            if node.turn % 2 == 0:
                board = node.board
                prev_move = game.rot_move(node.prev_move)
                moves = node.moves
            else:
                board = -np.flip(node.board)
                prev_move = node.prev_move
                moves = game.rot_moves(node.moves)
            if prev_move == game.MOVE_EMPTY:
                prev_move = []
            game.print_board(board, prev_move, move)

            print('value : ', node.value, 'turn : ', node.turn)

            for i, move in enumerate(moves):
                print('[%.2d] %.2f %4.2f/%.4d' % (i, node.pos[i], node.total_q[i], node.visited[i]), ' ', move)
        else:
            print('NONE')
        key = input('?').lower()
        if key == '':
            pass
        elif key.isdigit():
            num = int(key)
            move = moves[num]
            
        elif key[0] == 'g':
            num = int(key[1:])
            node = node.children[num]
            history.append(node)
            move = []
        elif key == 'b':
            if len(history) > 1:
                del history[-1]
                node = history[-1]
        elif key == 'x' or key == 'q':
            if len(key) > 1:
                return int(key[1:])                
            else:
                return -1
        





if __name__ == '__main__':
    board = game.get_init_board(0, 0)
    mcts = Mcts(board)
    while True:
        mcts.travel(100)
        n = explore(mcts)
        if n == -2:
            break
        mcts.move(n)

