//! Максимально оптимизированная BPE Training Implementation
//!
//! Ключевые оптимизации:
//! 1. Flat corpus - один Vec<u32> вместо HashMap<Vec<u32>, usize>
//! 2. Linked list на индексах - O(1) merge без аллокаций
//! 3. Priority Queue для O(log N) поиска максимума
//! 4. Инкрементальное обновление только затронутых пар

use regex::Regex;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;

const DELETED: u32 = u32::MAX; // Маркер удаленного токена

/// Wrapper for priority queue - max heap by frequency
#[derive(Eq, PartialEq)]
struct PairFreq {
    pair: (u32, u32),
    count: usize,
}

impl Ord for PairFreq {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.count.cmp(&other.count)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for PairFreq {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Flat corpus representation with linked list
struct FlatCorpus {
    tokens: Vec<u32>,      // Все токены подряд
    next: Vec<usize>,      // Индекс следующего токена (linked list)
    prev: Vec<usize>,      // Индекс предыдущего токена
    pair_positions: HashMap<(u32, u32), HashSet<usize>>, // Позиции каждой пары (HashSet для O(1) удаления)
}

impl FlatCorpus {
    fn new(sequences: Vec<Vec<u32>>) -> Self {
        let total_len: usize = sequences.iter().map(|s| s.len()).sum();
        let mut tokens = Vec::with_capacity(total_len);
        let mut next = Vec::with_capacity(total_len);
        let mut prev = Vec::with_capacity(total_len);
        let mut pair_positions: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
        
        for seq in sequences {
            let start_idx = tokens.len();
            for (i, &token) in seq.iter().enumerate() {
                let idx = start_idx + i;
                tokens.push(token);
                
                if i == 0 {
                    prev.push(usize::MAX);
                } else {
                    prev.push(idx - 1);
                }
                
                if i == seq.len() - 1 {
                    next.push(usize::MAX);
                } else {
                    next.push(idx + 1);
                    // Записываем позицию пары в HashSet
                    let pair = (token, seq[i + 1]);
                    pair_positions.entry(pair).or_insert_with(HashSet::new).insert(idx);
                }
            }
        }
        
        FlatCorpus { tokens, next, prev, pair_positions }
    }
    
    /// Получить частоты пар из индекса
    fn get_pair_freqs(&self) -> HashMap<(u32, u32), usize> {
        self.pair_positions.iter()
            .map(|(&pair, positions)| (pair, positions.len()))
            .collect()
    }
    
    /// Merge пары в новый токен, возвращает затронутые позиции
    fn merge_pair(&mut self, pair: (u32, u32), new_token: u32) -> Vec<usize> {
        let positions = match self.pair_positions.remove(&pair) {
            Some(pos) => pos,
            None => return Vec::new(),
        };
        
        let mut affected_positions = Vec::new();
        
        for &i in &positions {
            // Проверяем что пара всё ещё валидна
            if self.tokens[i] == DELETED {
                continue;
            }
            
            let next_idx = self.next[i];
            if next_idx == usize::MAX || self.tokens[next_idx] == DELETED {
                continue;
            }
            
            if (self.tokens[i], self.tokens[next_idx]) != pair {
                continue;
            }
            
            // Удаляем старые пары из индекса - O(1) с HashSet
            if self.prev[i] != usize::MAX {
                let prev_idx = self.prev[i];
                if self.tokens[prev_idx] != DELETED {
                    let old_left_pair = (self.tokens[prev_idx], self.tokens[i]);
                    if let Some(positions) = self.pair_positions.get_mut(&old_left_pair) {
                        positions.remove(&prev_idx);
                        if positions.is_empty() {
                            self.pair_positions.remove(&old_left_pair);
                        }
                    }
                }
            }
            
            let next_next_idx = self.next[next_idx];
            if next_next_idx != usize::MAX && self.tokens[next_next_idx] != DELETED {
                let old_right_pair = (self.tokens[next_idx], self.tokens[next_next_idx]);
                if let Some(positions) = self.pair_positions.get_mut(&old_right_pair) {
                    positions.remove(&next_idx);
                    if positions.is_empty() {
                        self.pair_positions.remove(&old_right_pair);
                    }
                }
            }
            
            // Merge
            self.tokens[i] = new_token;
            self.tokens[next_idx] = DELETED;
            self.next[i] = next_next_idx;
            if next_next_idx != usize::MAX {
                self.prev[next_next_idx] = i;
            }
            
            // Добавляем новые пары в индекс - O(1) с HashSet
            if self.prev[i] != usize::MAX {
                let prev_idx = self.prev[i];
                if self.tokens[prev_idx] != DELETED {
                    let new_left_pair = (self.tokens[prev_idx], new_token);
                    self.pair_positions.entry(new_left_pair).or_insert_with(HashSet::new).insert(prev_idx);
                }
            }
            
            if next_next_idx != usize::MAX && self.tokens[next_next_idx] != DELETED {
                let new_right_pair = (new_token, self.tokens[next_next_idx]);
                self.pair_positions.entry(new_right_pair).or_insert_with(HashSet::new).insert(i);
            }
            
            affected_positions.push(i);
        }
        
        affected_positions
    }
}

/// Process a text chunk for parallel pre-tokenization
fn process_text_chunk(
    chunk_text: &str,
    regex: &Regex,
    special_tokens: &[String],
) -> Vec<Vec<u32>> {
    let mut sequences = Vec::new();
    
    if !special_tokens.is_empty() {
        // Split by special tokens
        let special_pattern = format!(
            "({})",
            special_tokens
                .iter()
                .map(|t| regex::escape(t))
                .collect::<Vec<_>>()
                .join("|")
        );
        let special_regex = Regex::new(&special_pattern).unwrap();
        
        let mut segments = Vec::new();
        let mut last_end = 0;
        
        for mat in special_regex.find_iter(chunk_text) {
            if mat.start() > last_end {
                segments.push(&chunk_text[last_end..mat.start()]);
            }
            last_end = mat.end();
        }
        if last_end < chunk_text.len() {
            segments.push(&chunk_text[last_end..]);
        }
        
        // Apply regex to each segment
        for segment in segments {
            for mat in regex.find_iter(segment) {
                let chunk = mat.as_str();
                let byte_vec: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
                if !byte_vec.is_empty() {
                    sequences.push(byte_vec);
                }
            }
        }
    } else {
        // No special tokens - just apply regex
        for mat in regex.find_iter(chunk_text) {
            let chunk = mat.as_str();
            let byte_vec: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
            if !byte_vec.is_empty() {
                sequences.push(byte_vec);
            }
        }
    }
    
    sequences
}

/// Find chunk boundaries in file based on special token
fn find_chunk_boundaries(
    file_path: &str,
    desired_num_chunks: usize,
    split_token: &str,
) -> Result<Vec<usize>, String> {
    let file = File::open(file_path)
        .map_err(|e| format!("Failed to open file: {}", e))?;
    
    let metadata = file.metadata()
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = metadata.len() as usize;
    
    let chunk_size = file_size / desired_num_chunks;
    let mut chunk_boundaries = vec![0];
    
    let split_bytes = split_token.as_bytes();
    let mini_chunk_size = 4096;
    
    for i in 1..desired_num_chunks {
        let initial_position = i * chunk_size;
        let mut reader = BufReader::new(File::open(file_path).unwrap());
        let mut buffer = vec![0u8; mini_chunk_size];
        let mut current_pos = initial_position;
        
        std::io::Seek::seek(&mut reader, std::io::SeekFrom::Start(current_pos as u64))
            .map_err(|e| format!("Seek failed: {}", e))?;
        
        loop {
            match std::io::Read::read(&mut reader, &mut buffer) {
                Ok(0) => {
                    chunk_boundaries.push(file_size);
                    break;
                }
                Ok(n) => {
                    if let Some(pos) = buffer[..n].windows(split_bytes.len())
                        .position(|window| window == split_bytes) {
                        chunk_boundaries.push(current_pos + pos);
                        break;
                    }
                    current_pos += n;
                }
                Err(_) => {
                    chunk_boundaries.push(file_size);
                    break;
                }
            }
        }
    }
    
    chunk_boundaries.push(file_size);
    chunk_boundaries.sort_unstable();
    chunk_boundaries.dedup();
    
    Ok(chunk_boundaries)
}

pub struct BpeTrainer {
    pub merges: HashMap<(u32, u32), u32>,
    pub vocab: HashMap<u32, Vec<u8>>,
    pub regex_pattern: String,
    pub special_tokens: HashMap<String, u32>,
}

impl BpeTrainer {
    pub fn new(regex_pattern: Option<String>) -> Self {
        let pattern = regex_pattern.unwrap_or_else(|| {
            // Упрощенный паттерн без lookahead для обычного regex
            r"'(?:[sdmt]|ll|ve|re)| ?[a-zA-Zа-яА-ЯёЁ]+| ?[0-9]+| ?[^\s\w]+|\s+".to_string()
        });
        
        BpeTrainer {
            merges: HashMap::new(),
            vocab: HashMap::new(),
            regex_pattern: pattern,
            special_tokens: HashMap::new(),
        }
    }
    
    pub fn train(
        &mut self,
        input_file_path: &str,
        vocab_size: usize,
        special_tokens: Vec<String>,
        boundary_split_token: &str,
        num_threads: usize,
        verbose: bool,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();
        
        if vocab_size < 256 {
            return Err("Vocab size must be >= 256".to_string());
        }
        
        let num_merges = vocab_size - 256;
        if verbose {
            println!("Training BPE tokenizer -> vocab_size: {}, num_merges: {}", vocab_size, num_merges);
        }
        
        // Pre-tokenization
        let pretokenize_start = std::time::Instant::now();
        let regex = Regex::new(&self.regex_pattern)
            .map_err(|e| format!("Invalid regex: {}", e))?;
        
        let sequences = if num_threads > 1 {
            if verbose {
                println!("Parallel pre-tokenization with {} threads", num_threads);
            }
            
            let boundaries = find_chunk_boundaries(input_file_path, num_threads, boundary_split_token)?;
            if verbose {
                println!("Created {} chunks", boundaries.len() - 1);
            }
            
            let file = File::open(input_file_path)
                .map_err(|e| format!("Failed to open file: {}", e))?;
            let mut reader = BufReader::new(file);
            
            let mut chunks = Vec::new();
            for i in 0..boundaries.len() - 1 {
                let start = boundaries[i];
                let end = boundaries[i + 1];
                let size = end - start;
                
                std::io::Seek::seek(&mut reader, std::io::SeekFrom::Start(start as u64))
                    .map_err(|e| format!("Seek failed: {}", e))?;
                
                let mut buffer = vec![0u8; size];
                std::io::Read::read_exact(&mut reader, &mut buffer)
                    .map_err(|e| format!("Read failed: {}", e))?;
                
                let chunk_text = String::from_utf8_lossy(&buffer).to_string();
                chunks.push(chunk_text);
            }
            
            if verbose {
                println!("Pre-tokenizing chunks...");
            }
            
            let results = Arc::new(Mutex::new(Vec::new()));
            let mut handles = vec![];
            
            for chunk in chunks {
                let regex_clone = Regex::new(&self.regex_pattern).unwrap();
                let special_tokens_clone = special_tokens.clone();
                let results_clone = Arc::clone(&results);
                
                let handle = thread::spawn(move || {
                    let chunk_seqs = process_text_chunk(&chunk, &regex_clone, &special_tokens_clone);
                    results_clone.lock().unwrap().push(chunk_seqs);
                });
                
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let mut all_sequences = Vec::new();
            for chunk_seqs in results.lock().unwrap().iter() {
                all_sequences.extend_from_slice(chunk_seqs);
            }
            
            all_sequences
        } else {
            if verbose {
                println!("Sequential pre-tokenization");
            }
            
            let file = File::open(input_file_path)
                .map_err(|e| format!("Failed to open file: {}", e))?;
            let mut reader = BufReader::new(file);
            let mut text = String::new();
            std::io::Read::read_to_string(&mut reader, &mut text)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            
            process_text_chunk(&text, &regex, &special_tokens)
        };
        
        if verbose {
            println!("Pre-tokenization time: {:.2}s", pretokenize_start.elapsed().as_secs_f64());
            println!("Total sequences: {}", sequences.len());
        }
        
        // Initialize vocab with bytes
        for idx in 0..256u32 {
            self.vocab.insert(idx, vec![idx as u8]);
        }
        
        // Build flat corpus
        if verbose {
            println!("Building flat corpus...");
        }
        let build_start = std::time::Instant::now();
        let mut corpus = FlatCorpus::new(sequences);
        if verbose {
            println!("Flat corpus built in {:.2}s", build_start.elapsed().as_secs_f64());
            println!("Total tokens: {}", corpus.tokens.len());
            println!("Pair positions index size: {}", corpus.pair_positions.len());
        }
        
        // Initial pair frequencies
        if verbose {
            println!("Calculating initial pair frequencies...");
        }
        let freq_start = std::time::Instant::now();
        let pair_freqs = corpus.get_pair_freqs();
        if verbose {
            println!("Pair frequencies calculated in {:.2}s", freq_start.elapsed().as_secs_f64());
            println!("Unique pairs: {}", pair_freqs.len());
        }
        
        // Build priority queue
        if verbose {
            println!("Building priority queue...");
        }
        let heap_start = std::time::Instant::now();
        let mut heap: BinaryHeap<PairFreq> = pair_freqs
            .iter()
            .map(|(&pair, &count)| PairFreq { pair, count })
            .collect();
        if verbose {
            println!("Priority queue built in {:.2}s", heap_start.elapsed().as_secs_f64());
        }
        
        println!("\nBPE merges: 0/{} (starting...)", num_merges);
        
        for i in 0..num_merges {
            // Логируем первые несколько merges для отладки
            if i < 5 || (i + 1) % 100 == 0 {
                let progress = (i + 1) as f64 / num_merges as f64 * 100.0;
                let elapsed = start_time.elapsed().as_secs_f64();
                println!("Starting merge {}/{} ({:.1}%) at {:.1}s", i + 1, num_merges, progress, elapsed);
            }
            
            let merge_start = std::time::Instant::now();
            
            // Find most frequent pair from heap - O(log N)
            let best_pair_freq = loop {
                match heap.pop() {
                    None => {
                        if verbose {
                            println!("No more pairs to merge after {} merges", i);
                        }
                        if verbose {
                            println!("Training time: {:.2} min", start_time.elapsed().as_secs_f64() / 60.0);
                        }
                        return Ok(());
                    }
                    Some(pf) => {
                        // Verify the count is still valid (not stale)
                        let current_count = corpus.pair_positions.get(&pf.pair).map(|v| v.len()).unwrap_or(0);
                        if current_count == pf.count && current_count > 0 {
                            break pf;
                        }
                        // Stale entry, continue
                    }
                }
            };
            
            let pair = best_pair_freq.pair;
            let pair_count = best_pair_freq.count;
            let idx = 256 + i as u32;
            
            if i < 5 {
                println!("  Found best pair: {:?} with count {}", pair, pair_count);
            }
            
            // Merge pair in corpus - O(K) где K = количество вхождений пары
            let merge_corpus_start = std::time::Instant::now();
            let _affected_positions = corpus.merge_pair(pair, idx);
            
            if i < 5 {
                println!("  Merge took {:.4}s", merge_corpus_start.elapsed().as_secs_f64());
            }
            
            // КРИТИЧНО: Добавляем новые/измененные пары в heap
            // Собираем все пары которые могли измениться
            let mut pairs_to_update = HashSet::new();
            
            // Новые пары с новым токеном
            for (&check_pair, positions) in &corpus.pair_positions {
                if check_pair.0 == idx || check_pair.1 == idx {
                    pairs_to_update.insert(check_pair);
                }
            }
            
            // Добавляем их в heap с актуальными частотами
            for &updated_pair in &pairs_to_update {
                if let Some(positions) = corpus.pair_positions.get(&updated_pair) {
                    let count = positions.len();
                    if count > 0 {
                        heap.push(PairFreq { pair: updated_pair, count });
                    }
                }
            }
            
            // Store merge
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&self.vocab[&pair.0]);
            bytes.extend_from_slice(&self.vocab[&pair.1]);
            self.vocab.insert(idx, bytes);
            self.merges.insert(pair, idx);
            
            // Progress logging
            if (i + 1) % 100 == 0 {
                let progress = (i + 1) as f64 / num_merges as f64 * 100.0;
                let elapsed = start_time.elapsed().as_secs_f64();
                let eta = elapsed / (i + 1) as f64 * (num_merges - i - 1) as f64;
                let active_pairs = corpus.pair_positions.len();
                
                println!("BPE merges: {}/{} ({:.1}%) | Elapsed: {:.1}s | ETA: {:.1}s | Active pairs: {}",
                    i + 1, num_merges, progress, elapsed, eta, active_pairs);
            }
            
            if verbose && (i + 1) % 1000 == 0 {
                println!("  Merge {}/{}: {:?} -> {} ({} occurrences) in {:.4}s",
                    i + 1, num_merges, pair, idx, pair_count, merge_start.elapsed().as_secs_f64());
            }
        }
        
        if verbose {
            println!("Training time: {:.2} min", start_time.elapsed().as_secs_f64() / 60.0);
        }
        
        Ok(())
    }
    
    pub fn register_special_tokens(&mut self, special_tokens: HashMap<String, u32>) {
        println!("Registered special tokens: {:?}", special_tokens);
        self.special_tokens = special_tokens;
    }
    
    pub fn save(&self, file_prefix: &str) -> Result<(), String> {
        let model_file = format!("{}.model", file_prefix);
        let mut file = File::create(&model_file)
            .map_err(|e| format!("Failed to create model file: {}", e))?;
        
        writeln!(file, "simple-bpe v1")
            .map_err(|e| format!("Write failed: {}", e))?;
        writeln!(file, "{}", self.regex_pattern)
            .map_err(|e| format!("Write failed: {}", e))?;
        writeln!(file, "{}", self.special_tokens.len())
            .map_err(|e| format!("Write failed: {}", e))?;
        
        for (special, idx) in &self.special_tokens {
            writeln!(file, "{} {}", special, idx)
                .map_err(|e| format!("Write failed: {}", e))?;
        }
        
        for (pair, _) in &self.merges {
            writeln!(file, "{} {}", pair.0, pair.1)
                .map_err(|e| format!("Write failed: {}", e))?;
        }
        
        let vocab_file = format!("{}.vocab", file_prefix);
        let mut file = File::create(&vocab_file)
            .map_err(|e| format!("Failed to create vocab file: {}", e))?;
        
        let mut inverted_merges = HashMap::new();
        for (pair, idx) in &self.merges {
            inverted_merges.insert(*idx, *pair);
        }
        
        for idx in 0..self.vocab.len() as u32 {
            if let Some(token) = self.vocab.get(&idx) {
                let s = String::from_utf8_lossy(token);
                if let Some(pair) = inverted_merges.get(&idx) {
                    let s0 = String::from_utf8_lossy(&self.vocab[&pair.0]);
                    let s1 = String::from_utf8_lossy(&self.vocab[&pair.1]);
                    writeln!(file, "[{}][{}] -> [{}] {}", s0, s1, s, idx)
                        .map_err(|e| format!("Write failed: {}", e))?;
                } else {
                    writeln!(file, "[{}] {}", s, idx)
                        .map_err(|e| format!("Write failed: {}", e))?;
                }
            }
        }
        
        for (special, idx) in &self.special_tokens {
            writeln!(file, "[{}] {}", special, idx)
                .map_err(|e| format!("Write failed: {}", e))?;
        }
        
        println!("✓ Tokenizer saved: {}, {}", model_file, vocab_file);
        Ok(())
    }
}
