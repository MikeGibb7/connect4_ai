import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import math

# --- Connect4 Game Environment ---
ROWS, COLS = 6, 7

def create_board():
    return np.zeros((ROWS, COLS), dtype=np.int32)

def is_valid_location(board, col):
    return board[0][col] == 0

def get_valid_locations(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]

def get_next_open_row(board, col):
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == 0:
            return r
    return None

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def check_win(board, piece):
    # horizontal
    for c in range(COLS-3):
        for r in range(ROWS):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    # vertical
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    # diagonal /
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True
    # diagonal \
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    return False

def isBoardFull(board):
    return not any(board[0, c] == 0 for c in range(COLS))

# --- Neural Net ---
class Connect4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(ROWS*COLS, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, COLS)

    def forward(self, x):
        x = x.view(-1, ROWS*COLS)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Board Evaluation ---
def count_n_in_a_row(board, player, n):
    count = 0
    ROWS, COLS = board.shape

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - n + 1):
            if all(board[r][c + i] == player for i in range(n)):
                count += 1

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - n + 1):
            if all(board[r + i][c] == player for i in range(n)):
                count += 1

    # Diagonal (top-left to bottom-right)
    for r in range(ROWS - n + 1):
        for c in range(COLS - n + 1):
            if all(board[r + i][c + i] == player for i in range(n)):
                count += 1

    # Diagonal (bottom-left to top-right)
    for r in range(n - 1, ROWS):
        for c in range(COLS - n + 1):
            if all(board[r - i][c + i] == player for i in range(n)):
                count += 1

    return count

def evaluate_board(board, player):
    reward = 0.0
    opponent = 2 if player == 1 else 1

    reward += 0.1 * count_n_in_a_row(board, player, 2)
    reward += 0.5 * count_n_in_a_row(board, player, 3)
    reward -= 0.5 * count_n_in_a_row(board, opponent, 3)

    middle_col = COLS // 2
    for r in range(ROWS):
        if board[r][middle_col] == player:
            reward += 0.01  

    if check_win(board, player):
        reward += 1.0
    if check_win(board, opponent):
        reward -= 1.0

    return reward

# --- Minimax Agent (Player 1) ---
def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = check_win(board, 1) or check_win(board, 2) or isBoardFull(board)

    if depth == 0 or is_terminal:
        if check_win(board, 1):
            return (None, 1000000)
        elif check_win(board, 2):
            return (None, -1000000)
        elif isBoardFull(board):
            return (None, 0)
        else:
            return (None, evaluate_board(board, 1))

    if maximizingPlayer:
        value = -math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 1)
            new_score = minimax(temp_board, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 2)
            new_score = minimax(temp_board, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

# --- Training with Minimax Opponent ---
def train_model(episodes=1000, gamma=0.9, minimax_depth=4):
    model = Connect4Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for ep in range(episodes):
        board = create_board()
        game_over = False
        player = 1  # minimax starts first

        while not game_over:
            if player == 1:
                # Minimax move
                col, _ = minimax(board, minimax_depth, -math.inf, math.inf, True)
            else:
                # RL move
                inp = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0)
                preds = model(inp)
                col = int(torch.argmax(preds))

                # exploration
                if random.random() < 0.2:
                    col = random.choice(get_valid_locations(board))

                if not is_valid_location(board, col):
                    game_over = True
                    continue

            # Apply move
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, player)

            # RL learns only on its own turns
            if player == 2:
                current_reward = evaluate_board(board, player)

                target = preds.clone()
                with torch.no_grad():
                    next_inp = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0)
                    next_preds = model(next_inp)
                    max_future = torch.max(next_preds)
                target[0, col] = current_reward + gamma * max_future

                loss = loss_fn(preds, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if check_win(board, player) or isBoardFull(board):
                game_over = True

            player = 2 if player == 1 else 1

        if ep % 50 == 0:
            print(f"Episode {ep}/{episodes}")

    torch.save(model.state_dict(), "connect4_ai.pth")
    print("✅ Training done vs minimax!")

if __name__ == "__main__":
    train_model()
