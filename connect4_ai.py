import sys
import numpy as np
import onnxruntime as ort
import os

ROWS, COLS = 6, 7

# Load ONNX model once
model_path = r"C:\Users\mikeg\githubProjects\connnect4_ai\connect4_ai\connect4_ai_minimax.onnx"
session = ort.InferenceSession(model_path)

def pick_move(board):
    board_np = np.array(board, dtype=np.float32).reshape(1, ROWS, COLS)
    inputs = {"board": board_np}
    outputs = session.run(None, inputs)
    move = int(np.argmax(outputs[0]))
    return move

if __name__ == "__main__":
    # Expect flat board from C++
    data = sys.stdin.read().strip().split()
    board = list(map(int, data))
    move = pick_move(board)
    print(move)
