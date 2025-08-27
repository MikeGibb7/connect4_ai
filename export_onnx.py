import torch
import torch.nn as nn

class Connect4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6*7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)  # output = 7 columns

    def forward(self, x):
        x = x.view(-1, 6*7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Create model and load trained weights
model = Connect4Net()
model.load_state_dict(torch.load("connect4_ai.pth"))
model.eval()

# Export to ONNX
dummy_input = torch.zeros(1, 6, 7)
torch.onnx.export(
    model,
    dummy_input,
    "connect4_ai_minimax.onnx",
    input_names=["board"],
    output_names=["move_scores"],
    dynamic_axes={"board": {0: "batch"}}
)

print("✅ Exported model to connect4_ai.onnx")
