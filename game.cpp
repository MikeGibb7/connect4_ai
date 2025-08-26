#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cstdio>
#include <array>
#include <sstream>
using namespace std;

const int ROWS = 6;
const int COLS = 7;
int board[ROWS][COLS] = {0};  // initialize everything to 0

// Function to display the game board
void displayBoard() {
    cout << "\n";
    // Display column numbers
    for (int i = 1; i <= COLS; i++) {
        cout << "  " << i << " ";
    }
    cout << "\n";
    
    // Display the board
    for (int i = 0; i < ROWS; i++) {
        cout << "|";
        for (int j = 0; j < COLS; j++) {
            if (board[i][j] == 0) {
                cout << "   ";
            } else if (board[i][j] == 1) {
                cout << " X ";
            } else {
                cout << " O ";
            }
            cout << "|";
        }
        cout << "\n";
    }
    
    // Display bottom border
    cout << "+";
    for (int i = 0; i < COLS; i++) {
        cout << "---+";
    }
    cout << "\n";
}

// Function to check if a column is full
bool isColumnFull(int col) {
    return board[0][col] != 0;
}

// Function to drop a piece in a column
bool dropPiece(int col, int player) {
    if (col < 0 || col >= COLS || isColumnFull(col)) {
        return false;
    }
    
    // Find the lowest empty position in the column
    for (int i = ROWS - 1; i >= 0; i--) {
        if (board[i][col] == 0) {
            board[i][col] = player;
            return true;
        }
    }
    return false;
}

// Function to check for horizontal win
bool checkHorizontal(int row, int col, int player) {
    int count = 0;
    // Check to the left
    for (int j = col; j >= 0 && board[row][j] == player; j--) {
        count++;
    }
    // Check to the right
    for (int j = col + 1; j < COLS && board[row][j] == player; j++) {
        count++;
    }
    return count >= 4;
}

// Function to check for vertical win
bool checkVertical(int row, int col, int player) {
    int count = 0;
    // Check downward
    for (int i = row; i < ROWS && board[i][col] == player; i++) {
        count++;
    }
    return count >= 4;
}

// Function to check for diagonal win (both directions)
bool checkDiagonal(int row, int col, int player) {
    int count = 0;
    
    // Check diagonal (top-left to bottom-right)
    for (int i = row, j = col; i >= 0 && j >= 0 && board[i][j] == player; i--, j--) {
        count++;
    }
    for (int i = row + 1, j = col + 1; i < ROWS && j < COLS && board[i][j] == player; i++, j++) {
        count++;
    }
    if (count >= 4) return true;
    
    // Check diagonal (top-right to bottom-left)
    count = 0;
    for (int i = row, j = col; i >= 0 && j < COLS && board[i][j] == player; i--, j++) {
        count++;
    }
    for (int i = row + 1, j = col - 1; i < ROWS && j >= 0 && board[i][j] == player; i++, j--) {
        count++;
    }
    return count >= 4;
}

// Function to check if the last move resulted in a win
bool checkWin(int row, int col, int player) {
    return checkHorizontal(row, col, player) || 
           checkVertical(row, col, player) || 
           checkDiagonal(row, col, player);
}

// Function to check if the board is full (draw)
bool isBoardFull() {
    for (int j = 0; j < COLS; j++) {
        if (!isColumnFull(j)) {
            return false;
        }
    }
    return true;
}

// Function to get the row where the last piece was placed
int getLastRow(int col) {
    for (int i = 0; i < ROWS; i++) {   // top → down
        if (board[i][col] != 0) {
            return i;  // first filled cell from the top is the "last placed"
        }
    }
    return -1;
}

// Function to reset the board
void resetBoard() {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            board[i][j] = 0;
        }
    }
}

void undoDrop(int col) {
    for (int i = 0; i < ROWS; i++) {   // start at top
        if (board[i][col] != 0) {
            board[i][col] = 0;  // remove the top-most filled cell
            break;
        }
    }
}

bool checkWinForPlayer(int player) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if (board[r][c] == player) {
                if (checkWin(r, c, player)) {
                    return true;
                }
            }
        }
    }
    return false;
}

int evaluateWindow(int window[4], int aiPlayer) {
    int score = 0;
    int oppPlayer = (aiPlayer == 1 ? 2 : 1);

    int aiCount = 0;
    int oppCount = 0;
    int emptyCount = 0;

    for (int i = 0; i < 4; i++) {
        if (window[i] == aiPlayer) aiCount++;
        else if (window[i] == oppPlayer) oppCount++;
        else emptyCount++;
    }

    if (aiCount == 4) score += 100000;
    else if (aiCount == 3 && emptyCount == 1) score += 50;
    else if (aiCount == 2 && emptyCount == 2) score += 10;

    if (oppCount == 3 && emptyCount == 1) score -= 80;
    if (oppCount == 2 && emptyCount == 2) score -= 15;

    return score;
}

int evaluateBoard(int aiPlayer) {
    // Immediate win/loss detection
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            if (board[r][c] != 0) {
                if (checkWin(r, c, 1)) return -100000;  // human win
                if (checkWin(r, c, 2)) return 100000;   // AI win
            }
        }
    }

    int score = 0;
    int window[4];

    // Center column preference
    for (int r = 0; r < ROWS; r++) {
        if (board[r][COLS / 2] == aiPlayer) {
            score += 5;
        }
    }

    // Horizontal windows
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c <= COLS - 4; c++) {
            for (int i = 0; i < 4; i++) window[i] = board[r][c+i];
            score += evaluateWindow(window, aiPlayer);
        }
    }

    // Vertical windows
    for (int c = 0; c < COLS; c++) {
        for (int r = 0; r <= ROWS - 4; r++) {
            for (int i = 0; i < 4; i++) window[i] = board[r+i][c];
            score += evaluateWindow(window, aiPlayer);
        }
    }

    // Diagonal (\)
    for (int r = 0; r <= ROWS - 4; r++) {
        for (int c = 0; c <= COLS - 4; c++) {
            for (int i = 0; i < 4; i++) window[i] = board[r+i][c+i];
            score += evaluateWindow(window, aiPlayer);
        }
    }

    // Diagonal (/)
    for (int r = 0; r <= ROWS - 4; r++) {
        for (int c = 3; c < COLS; c++) {
            for (int i = 0; i < 4; i++) window[i] = board[r+i][c-i];
            score += evaluateWindow(window, aiPlayer);
        }
    }

    return score;
}

int minimax(int depth, bool isMaximizing, int aiPlayer) {
    // if (checkWinForPlayer(1)) {
    //     cout << "DEBUG: Human already has a winning line!\n";
    // }
    // if (checkWinForPlayer(2)) {
    //     cout << "DEBUG: AI has a winning line!\n";
    // }
    if (checkWinForPlayer(1)) return -1000000;
    if (checkWinForPlayer(2)) return 1000000;

    // Evaluate terminal states
    if (depth == 0 || isBoardFull()) {
        return evaluateBoard(aiPlayer);
    }

    if (isMaximizing) {
        int bestScore = -1000000;
        for (int col = 0; col < COLS; col++) {
            if (!isColumnFull(col)) {
                dropPiece(col, aiPlayer);
                int score = minimax(depth - 1, false, aiPlayer);
                undoDrop(col);
                bestScore = max(bestScore, score);
            }
        }
        return bestScore;
    } else {
        int bestScore = 1000000;
        int opponent = (aiPlayer == 1 ? 2 : 1);
        for (int col = 0; col < COLS; col++) {
            if (!isColumnFull(col)) {
                dropPiece(col, opponent);
                int score = minimax(depth - 1, true, aiPlayer);
                undoDrop(col);
                bestScore = min(bestScore, score);
            }
        }
        return bestScore;
    }
}

int getONNXAIMove() {
    // Build command: echo board → python
    ostringstream cmd;
    cmd << "echo ";
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            cmd << board[r][c] << " ";
        }
    }
    cmd << "| python C:\\Users\\mikeg\\githubProjects\\connnect4_ai\\connect4_ai\\connect4_ai.py";

    // Open pipe to read Python's stdout
    FILE* pipe = _popen(cmd.str().c_str(), "r");
    if (!pipe) {
        cerr << "Error: could not open pipe to Python AI.\n";
        // fallback random move
        int aiMove;
        do {
            aiMove = rand() % COLS;
        } while (isColumnFull(aiMove));
        return aiMove;
    }

    int aiMove = -1;
    fscanf(pipe, "%d", &aiMove);
    _pclose(pipe);

    // Validate move
    if (aiMove < 0 || aiMove >= COLS || isColumnFull(aiMove)) {
        cerr << "Python AI gave invalid move. Falling back to random.\n";
        do {
            aiMove = rand() % COLS;
        } while (isColumnFull(aiMove));
    }

    return aiMove;
}


int main() {
    srand(time(0));
    int currentPlayer = 1; // Player 1 starts
    bool gameWon = false;
    bool gameDraw = false;
    int aiDifficulty;

    cout << "Welcome to Connect Four!\n";
    cout << "Player 1: X, Player 2: O\n";
    cout << "Enter AI difficulty (1-3): ";
    cin >> aiDifficulty;
    cout << "Enter column number (1-7) to drop your piece.\n";
    
    while (!gameWon && !gameDraw) {
        displayBoard();

        int column;

        if (currentPlayer == 1) {
            // Human move
            cout << "Player " << currentPlayer << "'s turn (X): ";
            cin >> column;
            column--; // convert to 0-based
            if (column < 0 || column >= COLS) {
                cout << "Invalid column! Please choose 1-7.\n";
                continue;
            }
            if (isColumnFull(column)) {
                cout << "Column is full! Choose another column.\n";
                continue;
            }
        } else {
            if(aiDifficulty == 1){
                // AI move
                do {
                    column = rand() % COLS;
                } while (isColumnFull(column));
                cout << "AI chooses column " << (column + 1) << "\n";
            } else if (aiDifficulty == 2) {
                int bestMove = -1;
                int bestScore = -1000000;

                for (int col = 0; col < COLS; col++) {
                    if (!isColumnFull(col)) {
                        dropPiece(col, 2); // simulate AI move
                        int score = minimax(5, false, 2); // depth = 5
                        undoDrop(col);

                        if (score > bestScore) {
                            bestScore = score;
                            bestMove = col;
                        }
                    }
                }
                // Fallback in case minimax fails
                if (bestMove == -1) {
                    // just pick the first available column
                    for (int col = 0; col < COLS; col++) {
                        if (!isColumnFull(col)) {
                            bestMove = col;
                            break;
                        }
                    }
                }

                column = bestMove; // <-- this is the move AI actually plays
                cout << "AI chooses column " << (column + 1) << "\n";
            }  else if (aiDifficulty == 3) {
                // ONNX AI
                column = getONNXAIMove();
                cout << "ONNX AI chooses column " << (column + 1) << "\n";
            }
        }

        // Drop the piece (only once!)
        dropPiece(column, currentPlayer);
        int row = getLastRow(column);

        // Check for win
        if (checkWin(row, column, currentPlayer)) {
            displayBoard();
            cout << "Player " << currentPlayer << " (" 
                << (currentPlayer == 1 ? "X" : "O") << ") wins!\n";
            gameWon = true;
        }
        // Check for draw
        else if (isBoardFull()) {
            displayBoard();
            cout << "It's a draw!\n";
            gameDraw = true;
        }
        else {
            // Switch players
            currentPlayer = (currentPlayer == 1) ? 2 : 1;
        }
    }
    
    // Ask if player wants to play again
    char playAgain;
    cout << "Would you like to play again? (y/n): ";
    cin >> playAgain;
    
    if (playAgain == 'y' || playAgain == 'Y') {
        resetBoard();
        main(); // Restart the game
    } else {
        cout << "Thanks for playing!\n";
    }
    
    return 0;
}