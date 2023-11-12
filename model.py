import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Read and parse the data from a file
with open('puzzles.txt', 'r') as file:
    puzzles_lines = file.readlines()

# Parse the FEN (Forsyth-Edwards Notation) and the move
puzzles_data = []
for line in puzzles_lines:
    i = 7  # Starting from the 8th line (index is 7 because indexing starts from 0)
    while i < len(puzzles_lines) - 2:
        # Assuming each chess record spans three lines: a title line, a FEN line, and a move line
        title_line = puzzles_lines[i].strip()
        fen_line = puzzles_lines[i + 1].strip()
        move_line = puzzles_lines[i + 2].strip()

        if move_line.startswith('1. '):  # Check if it's a line with chess moves
            moves = move_line.split(' ')[1]  # get the first move
            if 'x' in moves:
                moves = moves[0:1] + moves[2:]
            if '+' in moves:
                moves = moves[0:-1]
            puzzles_data.append((fen_line, moves))

        i += 3  # Jump to the start of the next chess record

# Create a DataFrame from the parsed data
df = pd.DataFrame(puzzles_data, columns=['FEN', 'Move'])

# Function to convert FEN to a matrix representation
def fen_to_matrix(fen):
    piece_to_value = {
        'p': 1, 'n': 2, 'b': 4, 'r': 4, 'q': 10, 'k': 20,
        'P': -1, 'N': -2, 'B': -4, 'R': -4, 'Q': -10, 'K': -20
    }
    matrix = []
    for char in fen.split(' ')[0]:
        if char.isdigit():
            matrix.extend([0] * int(char))  # Add empty spaces
        elif char.isalpha():
            if char not in piece_to_value:
                raise ValueError(f"Unexpected character '{char}' in FEN string '{fen}'")
            matrix.append(piece_to_value[char])  # Add pieces
    if len(matrix) != 64:
        raise ValueError(f"FEN string '{fen}' cannot be converted to an 8x8 matrix.")
    return matrix


# Function to extract parts of the FEN
def extract_fen_parts(fen):
    parts = fen.split(' ')
    return {
        'castling': parts[2],
        'en_passant': parts[3],
        'half_move_clock': int(parts[4]),
        'full_move_number': int(parts[5])
    }


# Apply transformations and concatenate results to the DataFrame
df = df.join(df['FEN'].apply(extract_fen_parts).apply(pd.Series))
df['Board'] = df['FEN'].apply(lambda fen: fen_to_matrix(fen.split(' ')[0]))


# Update the label encoding to encode the moving piece and destination square separately
def encode_move(move):
    # Split the move string into piece and destination
    piece, destination = move[0], move[-2:]
    return piece, destination

# Apply the encoding function
df[['Piece', 'Destination']] = df['Move'].apply(encode_move).tolist()

# Encode categorical features
label_encoders = {col: LabelEncoder() for col in ['Piece', 'Destination', 'castling', 'en_passant']}
for col, encoder in label_encoders.items():
    df[col + '_Label'] = encoder.fit_transform(df[col])

    if col == 'Piece':
        df[col + '_number'] = [ord(x) for x in df[col]]
    if col == 'Destination' or col == 'castling' or col == 'en_passant':
        value = []
        for x in df[col]:
            num = 0
            for ch in x:
                num += ord(ch)
            value.append(num)
        df[col + '_number'] = value

print(df)
# Prepare the features and labels
X_board = np.array(df['Board'].tolist())
X_additional = df[['half_move_clock', 'full_move_number', 'castling_number', 'en_passant_number',
                   'Piece_number', 'Destination_number']].values
X = np.hstack((X_board, X_additional))
y_piece = df['Piece_Label']
y_destination = df['Destination_Label']


# Split the dataset into training and testing sets
X_train, X_test, y_piece_train, y_piece_test, y_destination_train, y_destination_test \
    = train_test_split(X, y_piece, y_destination, test_size=0.2, random_state=3000)

# Initialize and train the MLPClassifiers
mlp_piece = MLPClassifier(hidden_layer_sizes=(64, 128), activation='relu', solver='adam', random_state=50000,
                          verbose=True, early_stopping=True, n_iter_no_change=20)
mlp_destination = MLPClassifier(hidden_layer_sizes=(128, 256, 512), activation='relu', solver='adam', random_state=39000,
                                verbose=True, early_stopping=True, n_iter_no_change=20)

# Train the models
mlp_piece.fit(X_train, y_piece_train)
mlp_destination.fit(X_train, y_destination_train)

# Make predictions
y_piece_pred = mlp_piece.predict(X_test)
y_destination_pred = mlp_destination.predict(X_test)

# Evaluate the models
accuracy_piece = accuracy_score(y_piece_test, y_piece_pred)
accuracy_destination = accuracy_score(y_destination_test, y_destination_pred)

print("Piece Classification Report:\n", classification_report(y_piece_test, y_piece_pred))
print("Destination Classification Report:\n", classification_report(y_destination_test, y_destination_pred))
print("Piece Confusion Matrix:\n", confusion_matrix(y_piece_test, y_piece_pred))
print("Destination Confusion Matrix:\n", confusion_matrix(y_destination_test, y_destination_pred))
print(f"Piece Accuracy: {accuracy_piece:.2%}")
print(f"Destination Accuracy: {accuracy_destination:.2%}")
joblib.dump(mlp_piece, 'mlp_piece_model.joblib')
joblib.dump(mlp_destination, 'mlp_destination_model.joblib')
