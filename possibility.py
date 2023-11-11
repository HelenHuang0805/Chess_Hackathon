from collections import defaultdict


class Possibility:
    def __init__(self):
        self.possibility = dict()
        self.move_counter = defaultdict(dict)
        self.piece_weight_counter = defaultdict(int)
        self.model = defaultdict(dict)

    def train(self, file_name):
        with open(file_name, encoding="utf8") as file:
            for line in file:
                parts = line.strip().split(' ')
                color = ""
                three_moves = []

                if 'w' in parts:
                    color = 'w'
                elif 'b' in parts:
                    color = 'b'
                elif line.startswith('1. '):
                    moves = line.split()

                    three_moves.append(moves[1])
                    if color == 'w':
                        three_moves.append(moves[2])
                    else:
                        three_moves.append(moves[2])
                    three_moves.append(moves[4])

                    # handle the three move
                    for move in three_moves:
                        weight = 0
                        piece = move[0:1]

                        # the piece is pawn
                        if piece.lower() == piece:
                            piece = "p"
                        else:
                            piece = piece.lower()

                        # when it eat the opponent piece
                        if 'x' in move:
                            weight += 2
                            next_pos = move[2:4]
                        else:
                            next_pos = move[1:3]

                        if '+' in move or '#' in move:
                            weight += 5

                        weight += 10

                        # add weight into dictionary
                        if next_pos not in self.move_counter[piece]:
                            self.move_counter[piece][next_pos] = 0
                        self.move_counter[piece][next_pos] += weight
                        self.piece_weight_counter[piece] += weight

        for piece, move_dict in self.move_counter.items():
            for next_pos, weight in move_dict.items():
                self.model[piece][next_pos] = self.move_counter[piece][next_pos] / self.piece_weight_counter[piece]

        return self.model
