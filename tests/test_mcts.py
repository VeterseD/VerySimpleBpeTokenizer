import chess
import numpy as np

from azchess.encoding import POLICY_SIZE
from azchess.mcts import SearchConfig, Tree, run_search


def uniform_evaluator(planes):
    return np.zeros((len(planes), POLICY_SIZE), np.float32), np.zeros(len(planes), np.float32)


def test_finds_mate_in_one_for_both_colors():
    for fen, mate in [
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
        ("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1"),
    ]:
        board = chess.Board(fen)
        tree = Tree()
        run_search([board], [tree], uniform_evaluator, 400, SearchConfig())
        best = tree.best_index()
        assert tree.root.moves[best].uci() == mate
        assert tree.root.W[best] / tree.root.N[best] > 0.99
        assert board.fen() == fen  # search must leave the board untouched


def test_batched_search_and_tree_reuse():
    boards = [chess.Board() for _ in range(3)]
    trees = [Tree(noise_rng=np.random.default_rng(i)) for i in range(3)]
    sims = run_search(boards, trees, uniform_evaluator, 50, SearchConfig())
    assert sims == 150
    for tree in trees:
        assert tree.root.N.sum() == 49  # first simulation expands the root
        visited = tree.best_index()
        tree.advance(visited)
        assert tree.root.expanded
