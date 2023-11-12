"""
The Brandeis Quant Club ML/AI Competition (November 2023)

Author: @Ephraim Zimmerman
Email: quants@brandeis.edu
Website: brandeisquantclub.com; quants.devpost.com

Description:

For any technical issues or questions please feel free to reach out to
the "on-call" hackathon support member via email at quants@brandeis.edu

Website/GitHub Repository:
You can find the latest updates, documentation, and additional resources for this project on the
official website or GitHub repository: https://github.com/EphraimJZimmerman/chess_hackathon_23

License:
This code is open-source and released under the MIT License. See the LICENSE file for details.
"""

import random
import chess
import time
from collections.abc import Iterator
from contextlib import contextmanager

import joblib
import numpy
import numpy as np

import test_bot


@contextmanager
def game_manager() -> Iterator[None]:
    """Creates context for game."""

    print("===== GAME STARTED =====")
    ping: float = time.perf_counter()
    try:
        # DO NOT EDIT. This will be replaced w/ judging context manager.
        yield
    finally:
        pong: float = time.perf_counter()
        total = pong - ping
        print(f"Total game time = {total:.3f} seconds")
    print("===== GAME ENDED =====")


class Bot:
    def __init__(self, fen=None):
        self.board = chess.Board(fen if fen else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def check_move_is_legal(self, initial_position, new_position) -> bool:
        """
            To check if, from an initial position, the new position is valid.

            Args:
                initial_position (str): The starting position given chess notation.
                new_position (str): The new position given chess notation.

            Returns:
                bool: If this move is legal
        """

        return chess.Move.from_uci(initial_position + new_position) in self.board.legal_moves

    def next_move(self) -> str:
        """
            The main call and response loop for playing a game of chess.

            Returns:
                str: The current location and the next move.
        """

        # Assume that you are playing an arbitrary game. This function, which is
        # the core "brain" of the bot, should return the next move in any circumstance.

        move = str(random.choice([_ for _ in self.board.legal_moves]))
        print("My move: " + move)
        return move


# Add promotion stuff
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
            matrix.append(piece_to_value[char])  # Add pieces
    if len(matrix) != 64:
        raise ValueError(f"FEN string '{fen}' cannot be converted to a 8x8 matrix.")
    return matrix


def extract_fen_parts(fen):
    parts = fen.split(' ')
    return {
        'castling': parts[2],
        'en_passant': parts[3],
        'half_move_clock': int(parts[4]),
        'full_move_number': int(parts[5])
    }


def get_next_move_from_poss(model):
    next_move_max_poss = 0.0
    future_move = None

    for move in available_next_moves:
        next_square = move.uci()[2:4]
        current_square = chess.parse_square(move.uci()[0:2])
        piece = chess_bot.board.piece_at(current_square)

        lower_case_piece = str(piece).lower()
        if next_square not in model[lower_case_piece]:
            continue

        next_poss = model[lower_case_piece][next_square]

        if next_poss > next_move_max_poss:
            future_move = move.uci()
            next_move_max_poss = next_poss
    return future_move


def get_next_move_from_network(mlp_piece, mlp_destination):
    # convert the current fen to matrix
    fen = chess_bot.board.fen()
    board_matrix = np.array(fen_to_matrix(fen))
    predict_strength = dict()

    for move in available_next_moves:
        destination = move.uci()[2:4]
        des_int = 0
        for x in destination:
            des_int += ord(x)
        current_square = chess.parse_square(move.uci()[0:2])
        piece = chess_bot.board.piece_at(current_square)
        piece_number = ord(str(piece))

        # addition part for predicting
        addition_part = extract_fen_parts(fen)
        addition_part_arr = []
        for key, value in addition_part.items():
            if key == 'castling' or key == 'en_passant':
                num = 0
                for ch in value:
                    num += ord(ch)
                addition_part_arr.append(num)
            else:
                addition_part_arr.append(value)
        addition_part_arr.append(piece_number)
        addition_part_arr.append(des_int)

        predict_data = np.hstack((board_matrix, addition_part_arr))
        predict_data_2d = np.array([predict_data])

        # predict
        piece_pos = mlp_piece.predict(predict_data_2d)
        des_pos = mlp_destination.predict(predict_data_2d)

        predict_strength[(str(piece), str(move))] = piece_pos*des_pos

    max_strength = max(predict_strength.values())
    max_moves = set()

    for key, value in predict_strength.items():
        if value == max_strength:
            max_moves.add(key)

    return max_moves, max_strength


if __name__ == "__main__":
    win_count = 0
    drawn_count = 0
    lose_count = 0
    total_game = 100
    count_game = 0

    mlp_piece = joblib.load('mlp_piece_model.joblib')
    mlp_destination = joblib.load('mlp_destination_model.joblib')

    for i in range(0, total_game):
        count_game += 1
        chess_bot = Bot()  # you can enter a FEN here, like Bot("...")
        with game_manager():

            """
            
            Feel free to make any adjustments as you see fit. The desired outcome 
            is to generate the next best move, regardless of whether the bot 
            is controlling the white or black pieces. The code snippet below 
            serves as a useful testing framework from which you can begin 
            developing your strategy.
    
            """

            playing = True
            while playing:
                if chess_bot.board.turn:
                    available_next_moves = chess_bot.board.legal_moves

                    # get move form network
                    max_moves, max_strength = get_next_move_from_network(mlp_piece, mlp_destination)
                    # get move from possibility
                    next_move_from_poss = get_next_move_from_poss(model)

                    if len(max_moves) == 1:
                        chess_bot.board.push_san(max_moves.pop()[1])
                    else:
                        chess_move = list(max_moves)[random.randint(0, len(max_moves)-1)][1]
                        print("AI Move", chess_move)
                        chess_bot.board.push_san(chess_move)

                else:
                    chess_bot.board.push_san(chess_bot.next_move())
                print(chess_bot.board, end="\n\n")

                if chess_bot.board.is_game_over():
                    if chess_bot.board.is_stalemate():
                        print("Is stalemate")
                    elif chess_bot.board.is_insufficient_material():
                        print("Is insufficient material")

                    # EX: Outcome(termination=<Termination.CHECKMATE: 1>, winner=True)
                    outcome = chess_bot.board.outcome()
                    print(outcome)
                    if outcome.winner is True:
                        win_count += 1
                    elif outcome.winner is None:
                        drawn_count += 1
                    playing = False

    print("win rate", win_count / total_game)
    print("drawn rate", drawn_count / total_game)
    print("game rounds", count_game)
