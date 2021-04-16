from game import *


class Node:
    def get_pos_value(board, moves):
        pos = np.ones(len(moves), dtype=np.float32) / len(moves)
        value = 1
        return pos, value
    def set_model(full_model):
        def func(board, moves):
            pos_row, value_row = full_model.predict(board[None, ...])
            pos = pos_row[0][[get_index(m) for m in moves]]
            pos = pos / np.sum(pos)
            value = value_row[0][0]
            return pos, value
        Node.get_pos_value = func

    def __init__(self, board, level, parent=None, dum=True, prev_move=MOVE_EMPTY):
        self.board = board
        
        self.level = level
        self.parent = parent
        self.dum = not parent.dum
        self.prev_move = prev_move

        self.finished = is_finished(board)
        if self.finished:
            self.moves = []
            self.pos = []
            self.value = self.finished
            self.children = []
        else:
            self.moves = get_all_moves(board)
            #get pos and value
            self.pos, self.value = Node.get_pos_value(board, moves)
            self.children = [None for _ in range(len(moves))]

        self.total_score = 0 # Q(s, a)
        self.score = 0 #W(s, a)
        self.visited = 0 #N(s, a)
        

    def deliver(self, index):
        next_board = get_next_board_rot(self.board, self.moves[i])
        child = Node(next_board, self.level + 1, self, not self.dum, self.moves[i])
        self.children[i] = child
        return child
    
    def get_next_node_index(self):
        q = [child.score for child in self.children]
        q = np.array(q, dtype=np.float32)

        u = 0.2 * self.pos * np.sqrt(self.visited) / (1 + children_visited)
        
        return np.argmax(q + u)

    def get_final_play(self):



class Mcts:
    def __init__(self, board):
        self.root = Node(board, parent=None, dum=True)

    def travel_once(self):
        node = self.root
        visited_nodes = []
        while True:
            visited_nodes.append(node)
            if node.finished or node.level > 150:
                break
            i = node.get_next_node_index()

            # 자식 노드 방문
            if node.children[i] != None:
                node = node.children[i]
                visited_nodes.append(node)
            # leaf node - 출산
            else:
                node = node.deliver(i)
                visited_nodes.append(node)
                break

        leaf_value = visited_nodes[-1].value
        for node in visited_nodes:
            node.visited += 1
            node.total_score += leaf_value
            node.score = node.total_score / node.visited
                
if __name__ == '__main__':
    Node.
    get_init_board()
    Mcts()