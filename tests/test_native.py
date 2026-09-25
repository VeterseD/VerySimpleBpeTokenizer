import io

import chess
import chess.pgn
import numpy as np
import pytest

import azchess_rs as rs
from azchess.encoding import POLICY_SIZE, encode_board, move_to_index
from azchess.inference import run_search


def uniform_evaluator(planes):
    return np.zeros((len(planes), POLICY_SIZE), np.float32), np.zeros(len(planes), np.float32)


def random_game(rng, max_plies):
    board = chess.Board()
    for _ in range(rng.integers(0, max_plies)):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(legal[rng.integers(len(legal))])
    return board


def check_matches_reference(fen, moves):
    board = chess.Board(fen) if fen else chess.Board()
    for m in moves:
        board.push_uci(m)
    np.testing.assert_array_equal(rs.encode_position(fen, moves)[0], encode_board(board))
    assert dict(rs.legal_moves(fen, moves)) == {m.uci(): move_to_index(m, board.turn) for m in board.legal_moves}


def test_native_matches_python_reference_on_random_games():
    rng = np.random.default_rng(0)
    for _ in range(300):
        board = random_game(rng, 200)
        check_matches_reference(None, [m.uci() for m in board.move_stack])


@pytest.mark.parametrize("fen, moves", [
    (None, ["g1f3", "g8f6", "f3g1", "f6g8"]),  # repetition plane
    ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", ["e1g1", "e8c8"]),  # castling encoded as king moves
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", []),  # en passant
    ("1n2k3/P7/8/8/8/8/6p1/4K2R b K - 0 1", []),  # black underpromotions
])
def test_native_matches_python_reference_special_cases(fen, moves):
    check_matches_reference(fen, moves)


def test_searcher_finds_mate_in_one_for_both_colors():
    searcher = rs.Searcher(num_slots=2, leaves_per_step=4)
    searcher.set_position(0, "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    searcher.set_position(1, "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    run_search(searcher, uniform_evaluator, 400)
    for slot, mate in ((0, "a1a8"), (1, "a8a1")):
        best, q, visits, pv = searcher.result(slot)
        assert best == mate and pv[0] == mate
        assert q > 0.99


def test_threefold_repetition_is_terminal():
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
    searcher = rs.Searcher()
    searcher.set_position(0, None, shuffle[:-1])
    assert searcher.game_over(0) is None
    searcher.set_position(0, None, shuffle)
    assert searcher.game_over(0) == 0.0


def test_selfplay_produces_consistent_training_data():
    sp = rs.SelfPlay(num_games=4, simulations=16, fast_simulations=4, full_search_prob=0.5, max_plies=40, seed=1)
    while sp.num_finished < 6:
        logits, values = uniform_evaluator(sp.collect())
        sp.apply(logits, values)
    planes, idx, probs, z, results, plies, movetext, terminations = sp.take_finished()
    assert planes.shape[1:] == (20, 8, 8) and len(planes) == len(idx) == len(probs) == len(z)
    assert 0 < len(z) < plies.sum()  # only full searches are recorded
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-5)
    assert (probs[idx < 0] == 0).all() and idx.max() < POLICY_SIZE
    assert set(np.abs(z)) <= {0.0, 1.0} and set(results) <= {-1.0, 0.0, 1.0}
    assert sp.total_sims > 0 and sp.num_finished == 0

    # PGN replays legally to the reported result and termination
    assert len(movetext) == len(results) == len(terminations)
    for text, result, n, termination in zip(movetext, results, plies, terminations):
        game = chess.pgn.read_game(io.StringIO(text))
        assert not game.errors
        board = game.end().board()
        assert len(board.move_stack) == n
        assert text.endswith({1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}[float(result)])
        expected = {
            "checkmate": board.is_checkmate(), "stalemate": board.is_stalemate(),
            "threefold repetition": board.is_repetition(3), "fifty-move rule": board.halfmove_clock >= 100,
            "insufficient material": board.is_insufficient_material(), "max plies": n == 40,
        }
        assert expected[termination], (termination, board.fen())
        comments = [node.comment for node in game.mainline() if node.comment]
        assert comments and all(c.startswith("N=") and " | " in c for c in comments)


def test_apply_rejects_wrong_batch_size():
    sp = rs.SelfPlay(num_games=2, seed=0)
    planes = sp.collect()
    logits, values = uniform_evaluator(planes[:1])
    with pytest.raises(ValueError):
        sp.apply(logits, values)
