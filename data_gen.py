import random
import copy
import json

EMPTY = 0
X = 1
O = 2

def check_winner(board):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a,b,c in wins:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return 0
    return None

def minimax(board, is_maximizing):
    winner = check_winner(board)
    if winner is not None:
        return {X: 1, O: -1, 0: 0}[winner], None

    best_score = float('-inf') if is_maximizing else float('inf')
    best_move = None

    player = X if is_maximizing else O

    for i in range(9):
        if board[i] == EMPTY:
            board[i] = player
            score, _ = minimax(board, not is_maximizing)
            board[i] = EMPTY

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = i
            else:
                if score < best_score:
                    best_score = score
                    best_move = i

    return best_score, best_move

def generate_random_board():
    board = [EMPTY] * 9
    moves = random.randint(2, 6)
    turn = X
    for _ in range(moves):
        empty_indices = [i for i in range(9) if board[i] == EMPTY]
        if not empty_indices or check_winner(board) is not None:
            break
        move = random.choice(empty_indices)
        board[move] = turn
        turn = O if turn == X else X
    return board

def get_optimal_move_vector(board):
    _, best_move = minimax(copy.deepcopy(board), is_maximizing=True)
    if best_move is None:
        return [0] * 9
    move_vector = [0] * 9
    move_vector[best_move] = 1
    return move_vector

def generate_dataset(n=1000):
    data = []
    for _ in range(n):
        while True:
            board = generate_random_board()
            if check_winner(board) is None:
                x_count = board.count(X)
                o_count = board.count(O)
                if x_count == o_count:
                    break
        move_vector = get_optimal_move_vector(board)
        data.append((board, move_vector))
    return data



