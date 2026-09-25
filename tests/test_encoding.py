import chess
import numpy as np

from azchess.encoding import NUM_PLANES, POLICY_SIZE, encode_board, move_to_index


def random_positions(n, seed=0):
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < n:
        board = chess.Board()
        for _ in range(rng.integers(0, 120)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[rng.integers(len(legal))])
        if any(board.legal_moves):
            positions.append(chess.Board(board.fen()))
    return positions


def test_legal_move_indices_unique_and_in_range():
    for board in random_positions(300) + [chess.Board("4k3/1P6/8/8/8/8/6p1/4K2R w K - 0 1")]:
        idx = [move_to_index(m, board.turn) for m in board.legal_moves]
        assert len(set(idx)) == len(idx)
        assert all(0 <= i < POLICY_SIZE for i in idx)


def test_encoding_is_color_symmetric():
    for board in random_positions(100, seed=1):
        mirrored = board.mirror()
        assert encode_board(board).shape == (NUM_PLANES, 8, 8)
        np.testing.assert_array_equal(encode_board(board), encode_board(mirrored))
        for move in board.legal_moves:
            m = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
            assert move_to_index(move, board.turn) == move_to_index(m, mirrored.turn)


def test_underpromotions_have_distinct_planes():
    board = chess.Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    idx = {move_to_index(m, board.turn) for m in board.legal_moves if m.from_square == chess.A7}
    assert len(idx) == 8  # a8 and xb8, each with 4 promotion pieces
