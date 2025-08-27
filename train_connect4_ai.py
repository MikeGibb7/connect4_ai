import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import math
from collections import deque

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

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

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

# --- DQN Training with Replay Buffer ---
def train_dqn(episodes=1000, gamma=0.95, batch_size=64, minimax_depth=3):
    policy_net = Connect4Net()
    target_net = Connect4Net()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer()

    epsilon = 1.0  # exploration rate
    epsilon_min = 0.1
    epsilon_decay = 0.995

    for ep in range(episodes):
        board = create_board()
        game_over = False
        player = 1  # minimax starts

        while not game_over:
            if player == 1:
                col, _ = minimax(board, minimax_depth, -math.inf, math.inf, True)
            else:
                state = board.flatten().astype(np.float32)

                if random.random() < epsilon:
                    col = random.choice(get_valid_locations(board))
                else:
                    with torch.no_grad():
                        q_values = policy_net(torch.tensor(state).unsqueeze(0))
                        col = int(torch.argmax(q_values))

                if not is_valid_location(board, col):
                    col = random.choice(get_valid_locations(board))

                row = get_next_open_row(board, col)
                drop_piece(board, row, col, 2)

                reward = evaluate_board(board, 2)
                next_state = board.flatten().astype(np.float32)
                done = check_win(board, 2) or check_win(board, 1) or isBoardFull(board)

                buffer.push(state, col, reward, next_state, done)

                if len(buffer) > batch_size:
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                    q_values = policy_net(states)
                    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                    with torch.no_grad():
                        next_q_values = target_net(next_states).max(1)[0]
                        expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                    loss = loss_fn(state_action_values, expected_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if check_win(board, player) or isBoardFull(board):
                game_over = True

            player = 2 if player == 1 else 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 20 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {ep}/{episodes}, epsilon={epsilon:.3f}")

    torch.save(policy_net.state_dict(), "connect4_dqn.pth")
    print("✅ DQN Training done!")

if __name__ == "__main__":
    train_dqn()
