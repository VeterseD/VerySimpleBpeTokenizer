"""Board and move encoding in the AlphaZero layout, always from the side to move."""
from __future__ import annotations

from functools import lru_cache

import chess
import numpy as np

# 0-5 our P N B R Q K, 6-11 theirs, 12 repetition, 13-16 castling (us K/Q, them K/Q),
# 17 halfmove clock (raw, scaled in the model), 18 en passant square, 19 ones.
NUM_PLANES = 20
HALFMOVE_PLANE = 17
POLICY_PLANES = 73
POLICY_SIZE = POLICY_PLANES * 64
MAX_LEGAL_MOVES = 256  # chess maximum is 218

_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def _bb_planes(bbs: list[int]) -> np.ndarray:
    raw = np.array(bbs, dtype="<u8").view(np.uint8)
    return np.unpackbits(raw, bitorder="little").reshape(len(bbs), 8, 8)


def encode_board(board: chess.Board) -> np.ndarray:
    """Returns uint8 planes of shape (NUM_PLANES, 8, 8), indexed [plane, rank, file]."""
    us, them = board.turn, not board.turn
    bbs = [board.pieces_mask(pt, us) for pt in _PIECES] + [board.pieces_mask(pt, them) for pt in _PIECES]
    ep = board.ep_square if board.has_legal_en_passant() else None
    bbs.append(chess.BB_SQUARES[ep] if ep is not None else 0)
    if us == chess.BLACK:
        bbs = [chess.flip_vertical(bb) for bb in bbs]

    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.uint8)
    bits = _bb_planes(bbs)
    planes[:12] = bits[:12]
    planes[18] = bits[12]
    planes[12] = board.is_repetition(2)
    planes[13] = board.has_kingside_castling_rights(us)
    planes[14] = board.has_queenside_castling_rights(us)
    planes[15] = board.has_kingside_castling_rights(them)
    planes[16] = board.has_queenside_castling_rights(them)
    planes[HALFMOVE_PLANE] = min(board.halfmove_clock, 255)
    planes[19] = 1
    return planes


# (dfile, drank): N, NE, E, SE, S, SW, W, NW
_QUEEN_DIRS = {d: i for i, d in enumerate([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])}
_KNIGHT_DIRS = {d: i for i, d in enumerate([(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)])}
_UNDERPROMOTIONS = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


@lru_cache(maxsize=None)
def _index(from_sq: int, to_sq: int, promotion: int | None) -> int:
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    if promotion in _UNDERPROMOTIONS:
        plane = 64 + 3 * _UNDERPROMOTIONS[promotion] + (df + 1)
    elif (df, dr) in _KNIGHT_DIRS:
        plane = 56 + _KNIGHT_DIRS[(df, dr)]
    else:
        dist = max(abs(df), abs(dr))
        direction = ((df > 0) - (df < 0), (dr > 0) - (dr < 0))
        plane = _QUEEN_DIRS[direction] * 7 + dist - 1
    return plane * 64 + from_sq


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Policy index (plane * 64 + from_square) matching the (73, 8, 8) policy head output."""
    from_sq, to_sq = move.from_square, move.to_square
    if turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    return _index(from_sq, to_sq, move.promotion)
