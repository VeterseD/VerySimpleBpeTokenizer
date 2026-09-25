//! Native core: board encoding, legal move generation (shakmaty) and batched PUCT MCTS.
//! Python only runs the network: `collect()` returns leaf planes, `apply()` takes logits/values.

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{
    CastlingMode, CastlingSide, Chess, Color, EnPassantMode, File, Move, Piece, Position, Role, Square,
};

pub const NUM_PLANES: usize = 20;
pub const PLANE_SIZE: usize = NUM_PLANES * 64;
pub const POLICY_SIZE: usize = 73 * 64;
pub const MAX_LEGAL_MOVES: usize = 256;
const HALFMOVE_PLANE: usize = 17;

// ---------------------------------------------------------------- rng

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Marsaglia-Tsang.
    fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            let u = self.uniform();
            return self.gamma(alpha + 1.0) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            if self.uniform().ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

// ---------------------------------------------------------------- encoding

const ROLES: [Role; 6] = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King];

fn hash(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

fn fill_bb(out: &mut [u8], plane: usize, bb: u64, flip: bool) {
    let mut b = if flip { bb.swap_bytes() } else { bb };
    while b != 0 {
        let sq = b.trailing_zeros() as usize;
        out[plane * 64 + sq] = 1;
        b &= b - 1;
    }
}

fn fill_plane(out: &mut [u8], plane: usize, value: u8) {
    out[plane * 64..(plane + 1) * 64].fill(value);
}

/// Same layout as azchess/encoding.py: side to move at the bottom, [plane, rank, file].
fn encode(pos: &Chess, repetition: bool, out: &mut [u8]) {
    out.fill(0);
    let us = pos.turn();
    let them = us.other();
    let flip = us == Color::Black;
    let board = pos.board();
    for (k, &role) in ROLES.iter().enumerate() {
        fill_bb(out, k, board.by_piece(Piece { color: us, role }).into(), flip);
        fill_bb(out, 6 + k, board.by_piece(Piece { color: them, role }).into(), flip);
    }
    fill_plane(out, 12, repetition as u8);
    let castles = pos.castles();
    fill_plane(out, 13, castles.has(us, CastlingSide::KingSide) as u8);
    fill_plane(out, 14, castles.has(us, CastlingSide::QueenSide) as u8);
    fill_plane(out, 15, castles.has(them, CastlingSide::KingSide) as u8);
    fill_plane(out, 16, castles.has(them, CastlingSide::QueenSide) as u8);
    fill_plane(out, HALFMOVE_PLANE, pos.halfmoves().min(255) as u8);
    if let Some(ep) = pos.ep_square(EnPassantMode::Legal) {
        fill_bb(out, 18, 1u64 << ep.to_u32(), flip);
    }
    fill_plane(out, 19, 1);
}

const QUEEN_DIRS: [(i32, i32); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
const KNIGHT_DIRS: [(i32, i32); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

fn move_index(m: Move, turn: Color) -> u16 {
    let (mut from, mut to, promotion) = match m {
        Move::Normal { from, to, promotion, .. } => (from, to, promotion),
        Move::EnPassant { from, to } => (from, to, None),
        Move::Castle { king, rook } => {
            let file = if rook.file() > king.file() { File::G } else { File::C };
            (king, Square::from_coords(file, king.rank()), None)
        }
        Move::Put { .. } => unreachable!("drops do not exist in standard chess"),
    };
    if turn == Color::Black {
        from = from.flip_vertical();
        to = to.flip_vertical();
    }
    let df = to.file().to_u32() as i32 - from.file().to_u32() as i32;
    let dr = to.rank().to_u32() as i32 - from.rank().to_u32() as i32;
    let under = match promotion {
        Some(Role::Knight) => Some(0),
        Some(Role::Bishop) => Some(1),
        Some(Role::Rook) => Some(2),
        _ => None,
    };
    let plane = if let Some(u) = under {
        64 + 3 * u + (df + 1)
    } else if let Some(k) = KNIGHT_DIRS.iter().position(|&d| d == (df, dr)) {
        56 + k as i32
    } else {
        let dist = df.abs().max(dr.abs());
        let dir = QUEEN_DIRS.iter().position(|&d| d == (df.signum(), dr.signum())).unwrap() as i32;
        dir * 7 + dist - 1
    };
    (plane * 64 + from.to_u32() as i32) as u16
}

fn uci_string(m: Move) -> String {
    m.to_uci(CastlingMode::Standard).to_string()
}

/// Prior occurrences of the last position in `hist ++ path`, looking back at most `halfmoves` plies.
fn repetitions(hist: &[u64], path: &[u64], halfmoves: u32) -> u32 {
    let len = hist.len() + path.len();
    let get = |i: usize| if i < hist.len() { hist[i] } else { path[i - hist.len()] };
    let h = get(len - 1);
    let mut count = 0;
    let mut back = 2;
    while back <= halfmoves as usize && back < len {
        if get(len - 1 - back) == h {
            count += 1;
        }
        back += 2;
    }
    count
}

/// Game value for the side to move, or None if the game continues.
fn terminal_value(pos: &Chess, no_legal_moves: bool, reps: u32) -> Option<f32> {
    if no_legal_moves {
        return Some(if pos.is_check() { -1.0 } else { 0.0 });
    }
    if pos.halfmoves() >= 100 || reps >= 2 || pos.is_insufficient_material() {
        return Some(0.0);
    }
    None
}

fn parse_position(fen: Option<&str>, moves: &[String]) -> PyResult<(Chess, Vec<u64>)> {
    let err = |e: String| PyValueError::new_err(e);
    let mut pos = match fen {
        Some(f) => f
            .parse::<Fen>()
            .map_err(|e| err(format!("bad fen: {e}")))?
            .into_position::<Chess>(CastlingMode::Standard)
            .map_err(|e| err(format!("illegal fen: {e}")))?,
        None => Chess::default(),
    };
    let mut hist = vec![hash(&pos)];
    for s in moves {
        let m = s
            .parse::<UciMove>()
            .map_err(|e| err(format!("bad move {s}: {e}")))?
            .to_move(&pos)
            .map_err(|e| err(format!("illegal move {s}: {e}")))?;
        pos.play_unchecked(m);
        hist.push(hash(&pos));
    }
    Ok((pos, hist))
}

// ---------------------------------------------------------------- tree

#[derive(Clone, Copy)]
struct Edge {
    mv: Move,
    idx: u16,
    p: f32,
    n: u32,
    vl: u32,
    w: f32,
    child: u32, // 0 = not created; node 0 is always the root so it is never a child
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    New,
    Pending,
    Expanded,
    Terminal(f32),
}

#[derive(Clone)]
struct Node {
    edges: Vec<Edge>,
    state: State,
}

impl Node {
    fn new() -> Self {
        Node { edges: Vec::new(), state: State::New }
    }
}

#[derive(Clone, Copy)]
struct SearchParams {
    c_puct: f32,
    fpu_reduction: f32,
    leaves_per_step: usize,
}

struct Leaf {
    node: u32,
    path: Vec<(u32, u16)>,
    moves: Vec<(Move, u16)>,
    planes: Vec<u8>,
}

enum Descent {
    Terminal,
    Leaf(Leaf),
    Collision,
}

#[derive(Clone)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn new() -> Self {
        Tree { nodes: vec![Node::new()] }
    }

    fn root(&self) -> &Node {
        &self.nodes[0]
    }

    fn select(&self, node: u32, params: &SearchParams) -> u16 {
        let edges = &self.nodes[node as usize].edges;
        let (mut n_sum, mut w_sum) = (0u32, 0f32);
        for e in edges {
            n_sum += e.n + e.vl;
            w_sum += e.w - e.vl as f32;
        }
        let fpu = if n_sum > 0 { w_sum / n_sum as f32 - params.fpu_reduction } else { 0.0 };
        let explore = params.c_puct * ((n_sum + 1) as f32).sqrt();
        let mut best = (0u16, f32::NEG_INFINITY);
        for (i, e) in edges.iter().enumerate() {
            let n = e.n + e.vl;
            let q = if n > 0 { (e.w - e.vl as f32) / n as f32 } else { fpu };
            let score = q + explore * e.p / (1 + n) as f32;
            if score > best.1 {
                best = (i as u16, score);
            }
        }
        best.0
    }

    /// `value` is for the side to move at the leaf. Also removes the virtual loss taken on the way down.
    fn backprop(&mut self, path: &[(u32, u16)], mut value: f32) {
        for &(node, e) in path.iter().rev() {
            value = -value;
            let edge = &mut self.nodes[node as usize].edges[e as usize];
            edge.n += 1;
            edge.w += value;
            edge.vl -= 1;
        }
    }

    fn undo_virtual_loss(&mut self, path: &[(u32, u16)]) {
        for &(node, e) in path {
            self.nodes[node as usize].edges[e as usize].vl -= 1;
        }
    }

    fn descend(&mut self, root_pos: &Chess, hist: &[u64], params: &SearchParams) -> Descent {
        let mut pos = root_pos.clone();
        let mut path: Vec<(u32, u16)> = Vec::new();
        let mut hashes: Vec<u64> = Vec::new();
        let mut node = 0u32;
        loop {
            match self.nodes[node as usize].state {
                State::Expanded => {
                    let e = self.select(node, params);
                    path.push((node, e));
                    let (mv, child) = {
                        let edge = &mut self.nodes[node as usize].edges[e as usize];
                        edge.vl += 1;
                        (edge.mv, edge.child)
                    };
                    pos.play_unchecked(mv);
                    hashes.push(hash(&pos));
                    node = if child != 0 {
                        child
                    } else {
                        let id = self.nodes.len() as u32;
                        self.nodes.push(Node::new());
                        self.nodes[path.last().unwrap().0 as usize].edges[e as usize].child = id;
                        id
                    };
                }
                State::Terminal(v) => {
                    self.backprop(&path, v);
                    return Descent::Terminal;
                }
                State::Pending => {
                    self.undo_virtual_loss(&path);
                    return Descent::Collision;
                }
                State::New => {
                    let legal = pos.legal_moves();
                    let reps = repetitions(hist, &hashes, pos.halfmoves());
                    if let Some(v) = terminal_value(&pos, legal.is_empty(), reps) {
                        self.nodes[node as usize].state = State::Terminal(v);
                        self.backprop(&path, v);
                        return Descent::Terminal;
                    }
                    self.nodes[node as usize].state = State::Pending;
                    let turn = pos.turn();
                    let moves = legal.iter().map(|&m| (m, move_index(m, turn))).collect();
                    let mut planes = vec![0u8; PLANE_SIZE];
                    encode(&pos, reps >= 1, &mut planes);
                    return Descent::Leaf(Leaf { node, path, moves, planes });
                }
            }
        }
    }

    fn expand(&mut self, leaf: &Leaf, logits: &[f32]) {
        let max = leaf.moves.iter().map(|&(_, i)| logits[i as usize]).fold(f32::NEG_INFINITY, f32::max);
        let mut edges: Vec<Edge> = leaf
            .moves
            .iter()
            .map(|&(mv, idx)| Edge { mv, idx, p: (logits[idx as usize] - max).exp(), n: 0, vl: 0, w: 0.0, child: 0 })
            .collect();
        let sum: f32 = edges.iter().map(|e| e.p).sum();
        for e in &mut edges {
            e.p /= sum;
        }
        let node = &mut self.nodes[leaf.node as usize];
        node.edges = edges;
        node.state = State::Expanded;
    }

    fn add_noise(&mut self, rng: &mut Rng, alpha: f64, frac: f32) {
        let edges = &mut self.nodes[0].edges;
        let eta: Vec<f64> = edges.iter().map(|_| rng.gamma(alpha)).collect();
        let sum: f64 = eta.iter().sum::<f64>().max(1e-12);
        for (e, x) in edges.iter_mut().zip(eta) {
            e.p = (1.0 - frac) * e.p + frac * (x / sum) as f32;
        }
    }

    /// Keeps only the subtree under root edge `e`, renumbered so its root is node 0.
    fn advance(&mut self, e: usize) {
        let child = self.nodes[0].edges[e].child;
        if child == 0 {
            *self = Tree::new();
            return;
        }
        let mut old = std::mem::take(&mut self.nodes);
        let mut new_nodes = vec![std::mem::replace(&mut old[child as usize], Node::new())];
        let mut i = 0;
        while i < new_nodes.len() {
            for k in 0..new_nodes[i].edges.len() {
                let c = new_nodes[i].edges[k].child;
                if c != 0 {
                    new_nodes[i].edges[k].child = new_nodes.len() as u32;
                    new_nodes.push(std::mem::replace(&mut old[c as usize], Node::new()));
                }
            }
            i += 1;
        }
        self.nodes = new_nodes;
    }

    fn best_edge(&self) -> usize {
        let edges = &self.root().edges;
        (0..edges.len())
            .max_by(|&a, &b| (edges[a].n, edges[a].p).partial_cmp(&(edges[b].n, edges[b].p)).unwrap())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------- one searched position

struct Search {
    pos: Chess,
    hist: Vec<u64>, // hashes of every position so far, current last
    tree: Tree,
    sims_done: u32,
    target: u32,
    pending: Vec<Leaf>,
    noise: bool,
    noised: bool,
}

impl Search {
    fn new(pos: Chess, hist: Vec<u64>) -> Self {
        Search { pos, hist, tree: Tree::new(), sims_done: 0, target: 0, pending: Vec::new(), noise: false, noised: false }
    }

    fn root_reps(&self) -> u32 {
        repetitions(&self.hist, &[], self.pos.halfmoves())
    }

    fn collect(&mut self, params: &SearchParams, noise: Option<(&mut Rng, f64, f32)>) {
        if let Some((rng, alpha, frac)) = noise {
            if self.noise && !self.noised && self.tree.root().state == State::Expanded {
                self.tree.add_noise(rng, alpha, frac);
                self.noised = true;
            }
        }
        while self.sims_done < self.target && self.pending.len() < params.leaves_per_step {
            match self.tree.descend(&self.pos, &self.hist, params) {
                Descent::Terminal => self.sims_done += 1,
                Descent::Leaf(leaf) => {
                    self.sims_done += 1;
                    self.pending.push(leaf);
                }
                Descent::Collision => break,
            }
        }
    }

    fn apply(&mut self, logits: &[f32], values: &[f32]) {
        for (k, leaf) in std::mem::take(&mut self.pending).into_iter().enumerate() {
            self.tree.expand(&leaf, &logits[k * POLICY_SIZE..(k + 1) * POLICY_SIZE]);
            self.tree.backprop(&leaf.path, values[k]);
        }
    }

    fn needs_work(&self) -> bool {
        self.sims_done < self.target || !self.pending.is_empty()
    }

    fn play(&mut self, e: usize) {
        let mv = self.tree.root().edges[e].mv;
        self.pos.play_unchecked(mv);
        self.hist.push(hash(&self.pos));
        self.tree.advance(e);
        self.noised = false;
    }

    /// Value of the current (root) position for the side to move if the game is over.
    fn game_over(&self) -> Option<f32> {
        terminal_value(&self.pos, self.pos.legal_moves().is_empty(), self.root_reps())
    }
}

fn gather_planes(searches: &[&Search]) -> (Vec<u8>, usize) {
    let total: usize = searches.iter().map(|s| s.pending.len()).sum();
    let mut planes = Vec::with_capacity(total * PLANE_SIZE);
    for s in searches {
        for leaf in &s.pending {
            planes.extend_from_slice(&leaf.planes);
        }
    }
    (planes, total)
}

fn offsets(searches: &[&Search]) -> Vec<usize> {
    let mut acc = 0;
    searches
        .iter()
        .map(|s| {
            let o = acc;
            acc += s.pending.len();
            o
        })
        .collect()
}

fn check_batch(logits: &[f32], values: &[f32], expected: usize) -> PyResult<()> {
    if values.len() != expected || logits.len() != expected * POLICY_SIZE {
        return Err(PyValueError::new_err(format!(
            "expected {expected} evaluations, got logits {} values {}",
            logits.len() / POLICY_SIZE,
            values.len()
        )));
    }
    Ok(())
}

fn planes_array<'py>(py: Python<'py>, planes: Vec<u8>, n: usize) -> PyResult<Bound<'py, PyArray4<u8>>> {
    PyArray1::from_vec(py, planes).reshape([n, NUM_PLANES, 8, 8])
}

// ---------------------------------------------------------------- self-play

struct Sample {
    planes: Vec<u8>,
    idx: Vec<u16>,
    probs: Vec<f32>,
    turn: Color,
}

struct FinishedGame {
    samples: Vec<Sample>,
    result: f32, // white's perspective
    plies: u32,
}

#[derive(Clone, Copy)]
struct SelfPlayConfig {
    params: SearchParams,
    simulations: u32,
    fast_simulations: u32,
    full_search_prob: f64,
    temperature_plies: u32,
    max_plies: u32,
    dirichlet_alpha: f64,
    dirichlet_frac: f32,
}

struct Game {
    search: Search,
    samples: Vec<Sample>,
    plies: u32,
    full: bool,
    rng: Rng,
}

impl Game {
    fn new(cfg: &SelfPlayConfig, seed: u64) -> Self {
        let pos = Chess::default();
        let hist = vec![hash(&pos)];
        let mut g = Game { search: Search::new(pos, hist), samples: Vec::new(), plies: 0, full: true, rng: Rng(seed) };
        g.start_move(cfg);
        g
    }

    /// Playout cap randomization (KataGo): only full searches are recorded for training.
    fn start_move(&mut self, cfg: &SelfPlayConfig) {
        self.full = cfg.full_search_prob >= 1.0 || self.rng.uniform() < cfg.full_search_prob;
        self.search.target = if self.full { cfg.simulations } else { cfg.fast_simulations };
        self.search.sims_done = 0;
        self.search.noise = self.full;
    }

    fn collect(&mut self, cfg: &SelfPlayConfig) {
        self.search.collect(&cfg.params, Some((&mut self.rng, cfg.dirichlet_alpha, cfg.dirichlet_frac)));
    }

    fn after_apply(&mut self, cfg: &SelfPlayConfig) -> Option<FinishedGame> {
        if self.search.needs_work() {
            return None;
        }
        let root = self.search.tree.root();
        let visits: u32 = root.edges.iter().map(|e| e.n).sum();
        let probs: Vec<f32> = if visits > 0 {
            root.edges.iter().map(|e| e.n as f32 / visits as f32).collect()
        } else {
            root.edges.iter().map(|e| e.p).collect()
        };
        if self.full {
            let mut planes = vec![0u8; PLANE_SIZE];
            encode(&self.search.pos, self.search.root_reps() >= 1, &mut planes);
            self.samples.push(Sample {
                planes,
                idx: root.edges.iter().map(|e| e.idx).collect(),
                probs: probs.clone(),
                turn: self.search.pos.turn(),
            });
        }
        let e = if self.plies < cfg.temperature_plies {
            let mut r = self.rng.uniform() as f32;
            let mut chosen = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                if r < p {
                    chosen = i;
                    break;
                }
                r -= p;
            }
            chosen
        } else {
            self.search.tree.best_edge()
        };
        self.search.play(e);
        self.plies += 1;

        let result = match self.search.game_over() {
            Some(v) => Some(if self.search.pos.turn() == Color::White { v } else { -v }),
            None if self.plies >= cfg.max_plies => Some(0.0),
            None => None,
        };
        let finished = result.map(|result| {
            let samples = std::mem::take(&mut self.samples);
            let plies = self.plies;
            let seed = self.rng.next_u64();
            *self = Game::new(cfg, seed);
            FinishedGame { samples, result, plies }
        });
        if finished.is_none() {
            self.start_move(cfg);
        }
        finished
    }
}

/// Plays `num_games` games continuously; finished games are replaced by new ones.
#[pyclass(module = "azchess_rs")]
struct SelfPlay {
    games: Vec<Game>,
    cfg: SelfPlayConfig,
    finished: Vec<FinishedGame>,
    total_sims: u64,
}

#[pymethods]
impl SelfPlay {
    #[new]
    #[pyo3(signature = (
        num_games, simulations=200, fast_simulations=50, full_search_prob=0.25, temperature_plies=30,
        max_plies=400, c_puct=1.5, fpu_reduction=0.25, dirichlet_alpha=0.3, dirichlet_frac=0.25,
        leaves_per_step=1, seed=0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_games: usize,
        simulations: u32,
        fast_simulations: u32,
        full_search_prob: f64,
        temperature_plies: u32,
        max_plies: u32,
        c_puct: f32,
        fpu_reduction: f32,
        dirichlet_alpha: f64,
        dirichlet_frac: f32,
        leaves_per_step: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if simulations < 1 || fast_simulations < 1 || leaves_per_step < 1 || num_games < 1 {
            return Err(PyValueError::new_err("num_games, simulations and leaves_per_step must be >= 1"));
        }
        let cfg = SelfPlayConfig {
            params: SearchParams { c_puct, fpu_reduction, leaves_per_step },
            simulations,
            fast_simulations,
            full_search_prob,
            temperature_plies,
            max_plies,
            dirichlet_alpha,
            dirichlet_frac,
        };
        let mut seeder = Rng(seed ^ 0xA5A5_5A5A_1234_5678);
        let games = (0..num_games).map(|_| Game::new(&cfg, seeder.next_u64())).collect();
        Ok(SelfPlay { games, cfg, finished: Vec::new(), total_sims: 0 })
    }

    /// Runs tree descents on all games (in parallel) and returns the leaf positions to evaluate.
    fn collect<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let cfg = self.cfg;
        let games = &mut self.games;
        let (planes, n, sims) = py.detach(|| {
            let before: u64 = games.iter().map(|g| g.search.sims_done as u64).sum();
            games.par_iter_mut().for_each(|g| g.collect(&cfg));
            let after: u64 = games.iter().map(|g| g.search.sims_done as u64).sum();
            let searches: Vec<&Search> = games.iter().map(|g| &g.search).collect();
            let (planes, n) = gather_planes(&searches);
            (planes, n, after.saturating_sub(before))
        });
        self.total_sims += sims;
        planes_array(py, planes, n)
    }

    /// Feeds network outputs for the last `collect()` batch, backs them up and plays moves where search is done.
    fn apply(&mut self, py: Python<'_>, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let logits = logits.as_slice()?;
        let values = values.as_slice()?;
        let cfg = self.cfg;
        let games = &mut self.games;
        let offs = offsets(&games.iter().map(|g| &g.search).collect::<Vec<_>>());
        let expected = games.iter().map(|g| g.search.pending.len()).sum();
        check_batch(logits, values, expected)?;
        let done: Vec<FinishedGame> = py.detach(|| {
            games
                .par_iter_mut()
                .zip(offs)
                .filter_map(|(g, o)| {
                    g.search.apply(&logits[o * POLICY_SIZE..], &values[o..]);
                    g.after_apply(&cfg)
                })
                .collect()
        });
        self.finished.extend(done);
        Ok(())
    }

    #[getter]
    fn total_sims(&self) -> u64 {
        self.total_sims
    }

    #[getter]
    fn num_finished(&self) -> usize {
        self.finished.len()
    }

    /// Returns (planes u8 [T,20,8,8], policy_idx i16 [T,256] (-1 pad), policy_p f32 [T,256], z f32 [T],
    /// results f32 [games] (white's view), plies u32 [games]) and clears the finished list.
    #[allow(clippy::type_complexity)]
    fn take_finished<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray4<u8>>,
        Bound<'py, PyArray2<i16>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u32>>,
    )> {
        let games = std::mem::take(&mut self.finished);
        let t: usize = games.iter().map(|g| g.samples.len()).sum();
        let mut planes = Vec::with_capacity(t * PLANE_SIZE);
        let mut idx = vec![-1i16; t * MAX_LEGAL_MOVES];
        let mut probs = vec![0f32; t * MAX_LEGAL_MOVES];
        let mut z = Vec::with_capacity(t);
        let mut row = 0;
        for g in &games {
            for s in &g.samples {
                planes.extend_from_slice(&s.planes);
                for (k, (&i, &p)) in s.idx.iter().zip(&s.probs).enumerate() {
                    idx[row * MAX_LEGAL_MOVES + k] = i as i16;
                    probs[row * MAX_LEGAL_MOVES + k] = p;
                }
                z.push(if s.turn == Color::White { g.result } else { -g.result });
                row += 1;
            }
        }
        let results: Vec<f32> = games.iter().map(|g| g.result).collect();
        let plies: Vec<u32> = games.iter().map(|g| g.plies).collect();
        Ok((
            planes_array(py, planes, t)?,
            PyArray1::from_vec(py, idx).reshape([t, MAX_LEGAL_MOVES])?,
            PyArray1::from_vec(py, probs).reshape([t, MAX_LEGAL_MOVES])?,
            PyArray1::from_vec(py, z),
            PyArray1::from_vec(py, results),
            PyArray1::from_vec(py, plies),
        ))
    }
}

// ---------------------------------------------------------------- search for play (UCI / arena)

/// Independent searches on arbitrary positions, e.g. one per arena game or a single one for UCI.
#[pyclass(module = "azchess_rs")]
struct Searcher {
    slots: Vec<Search>,
    params: SearchParams,
}

impl Searcher {
    fn slot(&self, i: usize) -> PyResult<&Search> {
        self.slots.get(i).ok_or_else(|| PyValueError::new_err(format!("no slot {i}")))
    }
}

#[pymethods]
impl Searcher {
    #[new]
    #[pyo3(signature = (num_slots=1, c_puct=1.5, fpu_reduction=0.25, leaves_per_step=8))]
    fn new(num_slots: usize, c_puct: f32, fpu_reduction: f32, leaves_per_step: usize) -> Self {
        let slots = (0..num_slots).map(|_| Search::new(Chess::default(), vec![hash(&Chess::default())])).collect();
        Searcher { slots, params: SearchParams { c_puct, fpu_reduction, leaves_per_step: leaves_per_step.max(1) } }
    }

    /// Sets the position (startpos when fen is None) and clears the tree.
    #[pyo3(signature = (slot, fen=None, moves=Vec::new()))]
    fn set_position(&mut self, slot: usize, fen: Option<&str>, moves: Vec<String>) -> PyResult<()> {
        self.slot(slot)?;
        let (pos, hist) = parse_position(fen, &moves)?;
        self.slots[slot] = Search::new(pos, hist);
        Ok(())
    }

    /// Game value for the side to move in `slot` if the position is terminal.
    fn game_over(&self, slot: usize) -> PyResult<Option<f32>> {
        Ok(self.slot(slot)?.game_over())
    }

    /// Schedules `simulations` more simulations on every non-terminal slot listed (all if None).
    #[pyo3(signature = (simulations, slots=None))]
    fn start(&mut self, simulations: u32, slots: Option<Vec<usize>>) -> PyResult<()> {
        let which = slots.unwrap_or_else(|| (0..self.slots.len()).collect());
        for i in which {
            self.slot(i)?;
            let s = &mut self.slots[i];
            s.sims_done = 0;
            s.target = if s.game_over().is_some() { 0 } else { simulations };
        }
        Ok(())
    }

    /// Stops scheduling new simulations (pending leaves still need `apply`).
    fn stop(&mut self) {
        for s in &mut self.slots {
            s.target = s.sims_done;
        }
    }

    fn active(&self) -> bool {
        self.slots.iter().any(|s| s.needs_work())
    }

    fn collect<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let params = self.params;
        let slots = &mut self.slots;
        let (planes, n) = py.detach(|| {
            slots.par_iter_mut().for_each(|s| s.collect(&params, None));
            gather_planes(&slots.iter().collect::<Vec<_>>())
        });
        planes_array(py, planes, n)
    }

    fn apply(&mut self, py: Python<'_>, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let logits = logits.as_slice()?;
        let values = values.as_slice()?;
        let slots = &mut self.slots;
        let offs = offsets(&slots.iter().collect::<Vec<_>>());
        check_batch(logits, values, slots.iter().map(|s| s.pending.len()).sum())?;
        py.detach(|| {
            slots.par_iter_mut().zip(offs).for_each(|(s, o)| s.apply(&logits[o * POLICY_SIZE..], &values[o..]))
        });
        Ok(())
    }

    /// (best move uci, its Q for the side to move, root visits, principal variation) or None before any expansion.
    fn result(&self, slot: usize) -> PyResult<Option<(String, f32, u32, Vec<String>)>> {
        let s = self.slot(slot)?;
        let tree = &s.tree;
        if tree.root().state != State::Expanded {
            return Ok(None);
        }
        let best = tree.best_edge();
        let e = tree.root().edges[best];
        let q = if e.n > 0 { e.w / e.n as f32 } else { 0.0 };
        let visits = tree.root().edges.iter().map(|e| e.n).sum();
        let mut pv = Vec::new();
        let mut node = 0usize;
        while tree.nodes[node].state == State::Expanded && pv.len() < 20 {
            let edges = &tree.nodes[node].edges;
            let i = if node == 0 { best } else { (0..edges.len()).max_by_key(|&i| edges[i].n).unwrap() };
            if edges[i].n == 0 {
                break;
            }
            pv.push(uci_string(edges[i].mv));
            if edges[i].child == 0 {
                break;
            }
            node = edges[i].child as usize;
        }
        Ok(Some((uci_string(e.mv), q, visits, pv)))
    }
}

// ---------------------------------------------------------------- helpers for tests

/// Planes for a position, for checking against the Python reference encoder.
#[pyfunction]
#[pyo3(signature = (fen=None, moves=Vec::new()))]
fn encode_position<'py>(py: Python<'py>, fen: Option<&str>, moves: Vec<String>) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let (pos, hist) = parse_position(fen, &moves)?;
    let mut planes = vec![0u8; PLANE_SIZE];
    encode(&pos, repetitions(&hist, &[], pos.halfmoves()) >= 1, &mut planes);
    planes_array(py, planes, 1)
}

/// [(uci, policy index)] for all legal moves.
#[pyfunction]
#[pyo3(signature = (fen=None, moves=Vec::new()))]
fn legal_moves(fen: Option<&str>, moves: Vec<String>) -> PyResult<Vec<(String, u16)>> {
    let (pos, _) = parse_position(fen, &moves)?;
    Ok(pos.legal_moves().iter().map(|&m| (uci_string(m), move_index(m, pos.turn()))).collect())
}

#[pymodule]
fn azchess_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SelfPlay>()?;
    m.add_class::<Searcher>()?;
    m.add_function(wrap_pyfunction!(encode_position, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add("NUM_PLANES", NUM_PLANES)?;
    m.add("POLICY_SIZE", POLICY_SIZE)?;
    m.add("MAX_LEGAL_MOVES", MAX_LEGAL_MOVES)?;
    Ok(())
}
