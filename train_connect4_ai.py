import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# --- Connect4 Game Environment ---
ROWS, COLS = 6, 7

def create_board():
    return np.zeros((ROWS, COLS), dtype=np.int32)

def is_valid_location(board, col):
    return board[0][col] == 0

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
    # If the top row has any 0, the board is not full
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

def count_open_threes(board, player):
    """Count 3-in-a-row with at least one open side"""
    count = 0
    ROWS, COLS = board.shape

    # Horizontal open threes
    for r in range(ROWS):
        for c in range(COLS - 3):
            line = board[r, c:c+4]
            if list(line).count(player) == 3 and list(line).count(0) == 1:
                count += 1

    # Vertical open threes
    for c in range(COLS):
        for r in range(ROWS - 3):
            line = board[r:r+4, c]
            if list(line).count(player) == 3 and list(line).count(0) == 1:
                count += 1

    # Diagonal (top-left to bottom-right)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            line = [board[r+i][c+i] for i in range(4)]
            if line.count(player) == 3 and line.count(0) == 1:
                count += 1

    # Diagonal (bottom-left to top-right)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            line = [board[r-i][c+i] for i in range(4)]
            if line.count(player) == 3 and line.count(0) == 1:
                count += 1

    return count

def evaluate_board(board, player):
    reward = 0.0
    
    # Reward for each 2-in-a-row
    reward += 0.1 * count_n_in_a_row(board, player, 2)
    
    # Reward for each 3-in-a-row
    reward += 0.5 * count_n_in_a_row(board, player, 3)
    
    # Penalize if opponent has 3-in-a-row
    opponent = 2 if player == 1 else 1
    reward -= 0.5 * count_n_in_a_row(board, opponent, 3)
    
     # Penalize if opponent has "open threes" (3 in a row with space to complete 4)
    reward -= 0.3 * count_open_threes(board, opponent)
    middle_col = COLS // 2
    for r in range(ROWS):
        if board[r][middle_col] == player:
            reward += 0.02  # small bonus for controlling the center
    # Reward winning board
    if check_win(board, player):
        reward += 1.0
    if check_win(board, opponent):
        reward -= 1.0
        
    return reward

# --- Training (very simple self-play with random moves as baseline) ---
def train_model(episodes=10000, gamma=0.9, load_existing=False):
    model = Connect4Net()
    if load_existing:
        try:
            model.load_state_dict(torch.load("connect4_ai.pth"))
            print("✅ Loaded existing model for continued training.")
        except FileNotFoundError:
            print("⚠️ No existing model found, starting new model.")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for ep in range(episodes):
        board = create_board()
        game_over = False
        player = 1

        while not game_over:
            # Prepare input
            inp = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0)

            # Predict move scores
            preds = model(inp)
            col = int(torch.argmax(preds))  

            # Exploration: choose random column sometimes
            epsilon = max(0.05, 0.5 * (1 - ep / episodes))
            if random.random() < epsilon:
                col = random.choice([c for c in range(COLS) if is_valid_location(board, c)])

            if not is_valid_location(board, col):
                # Invalid move → end the game (or just pick random)
                game_over = True
                continue

            # Make the move
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, player)

            # Evaluate board for current player
            current_reward = evaluate_board(board, player)

            # Compute target scores
            target = preds.clone()
            # discounted future reward: max of predicted next state
            with torch.no_grad():
                next_inp = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0)
                next_preds = model(next_inp)
                max_future = torch.max(next_preds)
            target[0, col] = current_reward + gamma * max_future

            # Backpropagate
            loss = loss_fn(preds, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check terminal state
            if check_win(board, player) or isBoardFull(board):
                game_over = True

            # Switch player
            player = 2 if player == 1 else 1

        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes}")

    # Save and export
    torch.save(model.state_dict(), "connect4_ai.pth")
    dummy_input = torch.zeros(1, ROWS, COLS)
    torch.onnx.export(
        model,
        dummy_input,
        "connect4_ai.onnx",
        input_names=["board"],
        output_names=["move_scores"],
        dynamic_axes={"board": {0: "batch"}}
    )
    print("✅ Training done, model saved as connect4_ai.pth and connect4_ai.onnx")

if __name__ == "__main__":
    choice = input("Load existing AI for continued training? (y/n): ").lower()
    load_old = choice == 'y'
    train_model(episodes=10000, load_existing=load_old)