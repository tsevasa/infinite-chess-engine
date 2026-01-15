use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::Move;

use super::tt_defs::{TTFlag, TTProbeParams, TTStoreParams, value_from_tt, value_to_tt};
use super::{INFINITY, MATE_SCORE};

const ENTRIES_PER_BUCKET: usize = 4;
const NO_MOVE_SENTINEL: i64 = i64::MIN;

// ============================================================================
// Compact Move Representation
// ============================================================================

/// Compact move for TT storage. Uses i64 coordinates (required for infinite chess)
/// but removes Option overhead by using sentinel values.
///
/// Size: 40 bytes (from: 16 + to: 16 + piece_type: 1 + piece_color: 1 + promotion: 1 + padding: 5)
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TTMove {
    from_x: i64,
    from_y: i64,
    to_x: i64,
    to_y: i64,
    piece_type: u8,
    piece_color: u8,
    promotion: u8, // 0 = none, else PieceType as u8
}

impl TTMove {
    /// Create a "no move" sentinel
    #[inline]
    pub const fn none() -> Self {
        TTMove {
            from_x: NO_MOVE_SENTINEL,
            from_y: 0,
            to_x: 0,
            to_y: 0,
            piece_type: 0,
            piece_color: 0,
            promotion: 0,
        }
    }

    /// Check if this represents a valid move
    #[inline]
    pub fn is_some(&self) -> bool {
        self.from_x != NO_MOVE_SENTINEL
    }

    /// Create from a Move
    #[inline]
    pub fn from_move(m: &Move) -> Self {
        TTMove {
            from_x: m.from.x,
            from_y: m.from.y,
            to_x: m.to.x,
            to_y: m.to.y,
            piece_type: m.piece.piece_type() as u8,
            piece_color: m.piece.color() as u8,
            promotion: m.promotion.map_or(0, |p| p as u8),
        }
    }

    /// Convert back to Option<Move> for use in search
    #[inline]
    pub fn to_move(self) -> Option<Move> {
        if !self.is_some() {
            return None;
        }

        // Reconstruct piece type and color using safe conversion
        let piece_type = PieceType::from_u8(self.piece_type);
        let color = PlayerColor::from_u8(self.piece_color);
        let piece = Piece::new(piece_type, color);

        // Reconstruct promotion
        let promotion = if self.promotion == 0 {
            None
        } else {
            Some(unsafe { std::mem::transmute::<u8, PieceType>(self.promotion) })
        };

        Some(Move {
            from: Coordinate {
                x: self.from_x,
                y: self.from_y,
            },
            to: Coordinate {
                x: self.to_x,
                y: self.to_y,
            },
            piece,
            promotion,
            rook_coord: None, // TT doesn't store rook_coord; is_pseudo_legal validates
        })
    }

    /// Check if this TTMove matches a given Move (for move ordering)
    #[inline]
    #[allow(dead_code)]
    pub fn matches(&self, m: &Move) -> bool {
        self.is_some()
            && self.from_x == m.from.x
            && self.from_y == m.from.y
            && self.to_x == m.to.x
            && self.to_y == m.to.y
    }
}

// ============================================================================
// TT Entry (Single Entry in a Bucket)
// ============================================================================

/// Transposition Table entry - optimized for memory efficiency.
///
/// Layout (64 bytes total):
/// - key: u64         - Full hash key (8 bytes)
/// - score: i32       - Evaluation score (4 bytes)
/// - depth: u8        - Search depth (1 byte)
/// - gen_bound: u8    - Generation (6 bits) + Bound type (2 bits) (1 byte)
/// - padding: [u8; 10]- Padding to align to 64 bytes/cache line
/// - tt_move: TTMove  - Best move (40 bytes)
#[derive(Clone, Copy, Debug)]
#[repr(C)] // Ensure C layout for reliable size
pub struct TTEntry {
    /// Full 64-bit hash key for verification
    key: u64,
    /// Score from the search (with mate score adjustment for storage)
    score: i32,
    /// Static evaluation of the position
    eval: i32,
    /// Search depth that produced this result
    depth: u8,
    /// Packed: generation (upper 6 bits) + bound type (lower 2 bits)
    gen_bound: u8,
    /// Padding to reach 64 bytes (8+4+4+1+1+6+40 = 64)
    _padding: [u8; 6],
    /// Best move found (or sentinel for none)
    tt_move: TTMove,
}

impl TTEntry {
    /// Create an empty/invalid entry
    #[inline]
    pub const fn empty() -> Self {
        TTEntry {
            key: 0,
            score: 0,
            eval: INFINITY + 1, // Use INFINITY + 1 as "None"
            depth: 0,
            gen_bound: 0,
            _padding: [0; 6],
            tt_move: TTMove::none(),
        }
    }

    /// Check if entry is empty (never written to)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.gen_bound == 0 && self.key == 0
    }

    /// Extract the bound type from gen_bound
    #[inline]
    pub fn flag(&self) -> TTFlag {
        TTFlag::from_u8(self.gen_bound)
    }

    /// Extract the is_pv flag
    #[inline]
    pub fn is_pv(&self) -> bool {
        (self.gen_bound & 0x04) != 0
    }

    /// Extract the generation from gen_bound
    #[inline]
    pub fn generation(&self) -> u8 {
        self.gen_bound >> 3
    }

    /// Create packed gen_bound from generation, is_pv and flag
    #[inline]
    fn pack_gen_bound(generation: u8, is_pv: bool, flag: TTFlag) -> u8 {
        (generation << 3) | (if is_pv { 0x04 } else { 0 }) | (flag as u8)
    }

    /// Get the best move as Option<Move>
    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        self.tt_move.to_move()
    }
}

// ============================================================================
// TT Bucket (Cluster of Entries)
// ============================================================================

/// A bucket/cluster containing multiple TT entries.
/// This improves collision handling - when storing, we pick the least valuable
/// entry in the bucket to replace.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct TTBucket {
    entries: [TTEntry; ENTRIES_PER_BUCKET],
}

impl TTBucket {
    #[inline]
    pub const fn empty() -> Self {
        TTBucket {
            entries: [TTEntry::empty(); ENTRIES_PER_BUCKET],
        }
    }
}

// ============================================================================
// Transposition Table
// ============================================================================

/// Transposition Table with bucket-based collision handling.
///
/// Key optimizations:
/// - Bucket system: 4 entries per bucket reduces effective collision rate
/// - Align entries to cache lines (64 bytes)
/// - Store FULL 64-bit hash key to prevent collisions
/// - Power-of-two sizing for fast index calculation
use std::cell::UnsafeCell;

/// TranspositionTable with 0-overhead single-threaded interior mutability.
/// Uses UnsafeCell to allow mutation via &self, matching the shared TT API.
/// SAFE: This file is only compiled when multithreading is DISABLED.
pub struct LocalTranspositionTable {
    buckets: UnsafeCell<Vec<TTBucket>>,
    /// Bitmask for indexing (capacity is always power of two)
    mask: usize,
    /// Current generation (incremented each search)
    generation: UnsafeCell<u8>,
    /// Number of entries currently stored (for fill percentage)
    used: UnsafeCell<usize>,
}

// Required for OnceLock<LocalTranspositionTable>
// SAFETY: This implementation is only used in single-threaded builds (cfg check),
// where we guarantee only one thread accesses the TT at a time.
unsafe impl Sync for LocalTranspositionTable {}

impl LocalTranspositionTable {
    /// Create a new TT with approximately `size_mb` megabytes of storage.
    /// For WASM builds, the size is capped at 64MB to avoid browser memory limits.
    pub fn new(size_mb: usize) -> Self {
        // Cap size for WASM to stay within browser memory constraints
        #[cfg(target_arch = "wasm32")]
        let size_mb = size_mb.min(64);

        let bytes = size_mb.max(1) * 1024 * 1024;
        let bucket_size = std::mem::size_of::<TTBucket>();
        let num_buckets = (bytes / bucket_size).max(1);

        // Round down to power of two for efficient masking
        let mut cap_pow2 = 1usize;
        while cap_pow2 * 2 <= num_buckets {
            cap_pow2 *= 2;
        }

        LocalTranspositionTable {
            buckets: UnsafeCell::new(vec![TTBucket::empty(); cap_pow2]),
            mask: cap_pow2 - 1,
            generation: UnsafeCell::new(1), // Start at 1 so 0 indicates empty
            used: UnsafeCell::new(0),
        }
    }

    /// Get the hash for the current position
    #[inline]
    pub fn generate_hash(game: &GameState) -> u64 {
        game.hash
    }

    /// Number of buckets in the table
    #[inline]
    pub fn capacity(&self) -> usize {
        let buckets = unsafe { &*self.buckets.get() };
        buckets.len() * ENTRIES_PER_BUCKET
    }

    /// Number of entries currently stored
    #[inline]
    pub fn used_entries(&self) -> usize {
        unsafe { *self.used.get() }
    }

    /// Fill percentage in permille (0-1000)
    #[inline]
    pub fn fill_permille(&self) -> u32 {
        let capacity = self.capacity();
        if capacity == 0 {
            return 0;
        }
        ((self.used_entries() as u64 * 1000) / capacity as u64) as u32
    }

    /// Calculate bucket index from hash
    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Prefetch the TT bucket for the given hash into L1 cache.
    #[inline]
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    pub fn prefetch_entry(&self, hash: u64) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let idx = self.bucket_index(hash);
        let buckets = unsafe { &*self.buckets.get() };
        let ptr = buckets.as_ptr().wrapping_add(idx) as *const i8;
        // SAFETY: ptr points into a valid, allocated slice
        unsafe { _mm_prefetch(ptr, _MM_HINT_T0) };
    }

    /// Probe the TT for a position.
    pub fn probe(&self, params: &TTProbeParams) -> Option<(i32, i32, Option<Move>, bool)> {
        let hash = params.hash;
        let alpha = params.alpha;
        let beta = params.beta;
        let depth = params.depth;
        let ply = params.ply;
        let rule50_count = params.rule50_count;
        let rule_limit = params.rule_limit;

        let idx = self.bucket_index(hash);
        // SAFETY: Single-threaded build implied by module import
        let buckets = unsafe { &*self.buckets.get() };
        let bucket = &buckets[idx];

        // Search all entries in the bucket for a match
        for entry in &bucket.entries {
            // Check full 64-bit key or if empty
            if entry.key != hash || entry.is_empty() {
                continue;
            }

            // Found a matching entry
            let best_move = entry.best_move();

            // Only use score if depth is sufficient
            if entry.depth as usize >= depth {
                // Adjust score from TT to search value, handling 50-move rule
                let score = value_from_tt(entry.score, ply, rule50_count, rule_limit);

                // Check if we can use this score for a cutoff
                let usable_score = match entry.flag() {
                    TTFlag::Exact => Some(score),
                    TTFlag::LowerBound if score >= beta => Some(score),
                    TTFlag::UpperBound if score <= alpha => Some(score),
                    _ => None,
                };

                if let Some(s) = usable_score {
                    return Some((s, entry.eval, best_move, entry.is_pv()));
                }
            }

            // Depth insufficient or bounds don't allow cutoff, but move and evaluation are still useful
            return Some((INFINITY + 1, entry.eval, best_move, entry.is_pv()));
        }

        None
    }

    /// Probe the TT for Singular Extension data.
    pub fn probe_for_singular(
        &self,
        hash: u64,
        ply: usize,
    ) -> Option<(TTFlag, u8, i32, i32, Option<Move>, bool)> {
        let idx = self.bucket_index(hash);
        let buckets = unsafe { &*self.buckets.get() };
        let bucket = &buckets[idx];

        for entry in &bucket.entries {
            if entry.key != hash || entry.is_empty() {
                continue;
            }

            // Adjust mate scores for current ply
            let mut score = entry.score;
            if score > MATE_SCORE {
                score -= ply as i32;
            } else if score < -MATE_SCORE {
                score += ply as i32;
            }

            return Some((
                entry.flag(),
                entry.depth,
                score,
                entry.eval,
                entry.best_move(),
                entry.is_pv(),
            ));
        }

        None
    }

    /// Store an entry in the TT.
    /// Uses interior mutability via UnsafeCell.
    pub fn store(&self, params: &TTStoreParams) {
        let hash = params.hash;
        let depth = params.depth;
        let flag = params.flag;
        let score = params.score;
        let static_eval = params.static_eval;
        let is_pv = params.is_pv;
        let best_move = params.best_move;
        let ply = params.ply;

        // Adjust mate scores for storage
        let adjusted_score = value_to_tt(score, ply);

        let idx = self.bucket_index(hash);

        // SAFETY: Single-threaded build only
        let (generation, buckets, used_ptr) = unsafe {
            (
                *self.generation.get(),
                &mut *self.buckets.get(),
                self.used.get(),
            )
        };

        let bucket = &mut buckets[idx];

        // Find the best slot to use (search for existing entry or worst victim)
        let mut replace_idx = 0;
        let mut worst_score = i32::MAX;

        for (i, entry) in bucket.entries.iter_mut().enumerate() {
            // If we find our own position
            if entry.key == hash {
                // Preserve the existing move if no new move is provided:
                let move_to_store = if best_move.is_some() {
                    best_move.as_ref().map_or(TTMove::none(), TTMove::from_move)
                } else {
                    // Keep the old move
                    entry.tt_move
                };

                // Preserve the existing evaluation if no new evaluation is provided:
                let eval_to_store = if static_eval != INFINITY + 1 {
                    static_eval
                } else {
                    entry.eval
                };

                // Replacement condition for existing entries:
                // PV bonus = +2 for exact bounds, threshold = old_depth - 4
                let pv_bonus = if flag == TTFlag::Exact || is_pv { 2 } else { 0 };
                let new_adjusted_depth = depth as i32 + pv_bonus;
                let old_threshold = entry.depth as i32 - 4;

                // Replace if: exact bound, or new depth high enough, or entry is aged,
                // or if we're just updating the evaluation (DEPTH_UNSEARCHED)
                // 5-bit generation wrap check
                let relative_age = (generation.wrapping_sub(entry.generation())) & 0x1F;
                if flag == TTFlag::Exact
                    || new_adjusted_depth > old_threshold
                    || relative_age != 0
                    || depth == 0
                {
                    if entry.is_empty() {
                        unsafe { *used_ptr += 1 };
                    }
                    *entry = TTEntry {
                        key: hash,
                        depth: depth as u8,
                        gen_bound: TTEntry::pack_gen_bound(generation, is_pv, flag),
                        score: adjusted_score,
                        eval: eval_to_store,
                        _padding: [0; 6],
                        tt_move: move_to_store,
                    };
                } else if entry.depth >= 5 && entry.flag() != TTFlag::Exact {
                    // Apply soft aging to deep non-exact entries:
                    entry.depth = entry.depth.saturating_sub(1);
                }
                return;
            }

            // Calculate replacement priority score (lower = more replaceable)
            let entry_score = Self::calculate_replacement_score(entry, generation);

            if entry_score < worst_score {
                worst_score = entry_score;
                replace_idx = i;
            }
        }

        // Prepare the new entry
        let new_entry = TTEntry {
            key: hash,
            depth: depth as u8,
            gen_bound: TTEntry::pack_gen_bound(generation, is_pv, flag),
            score: adjusted_score,
            eval: static_eval,
            _padding: [0; 6],
            tt_move: best_move.as_ref().map_or(TTMove::none(), TTMove::from_move),
        };

        // Calculate value of the new entry
        let new_score = Self::calculate_replacement_score(&new_entry, generation);

        // Replace if new entry is more valuable OR victim is empty/aged
        if new_score >= worst_score {
            if bucket.entries[replace_idx].is_empty() {
                unsafe { *used_ptr += 1 };
            }
            bucket.entries[replace_idx] = new_entry;
        }
    }

    /// Calculate a score for replacement priority (higher = more valuable, less replaceable).
    ///
    /// Factors considered:
    /// - Empty entries are always replaceable (score i32::MIN)
    /// - Deeper entries are more valuable (Base value = depth)
    /// - Older entries are less valuable (Penalty = 2 * age_diff)
    /// - Exact/PV nodes get a small bonus
    #[inline]
    fn calculate_replacement_score(entry: &TTEntry, current_generation: u8) -> i32 {
        if entry.is_empty() {
            return i32::MIN;
        }

        let mut score = entry.depth as i32;

        // Age penalty: penalize 2 points per generation old
        // Use 5-bit generation difference (wrapping)
        let age_diff = (current_generation.wrapping_sub(entry.generation())) & 0x1F;
        score -= (age_diff as i32) * 2;

        // PV/Exact bonus: favor keeping entries from interesting nodes
        if entry.flag() == TTFlag::Exact || entry.is_pv() {
            score += 2;
        }

        score
    }

    /// Increment the generation counter (call at the start of each search from root)
    pub fn increment_age(&self) {
        let gen_ptr = self.generation.get();
        unsafe {
            // Wrap at 31 (5 bits)
            let val = (*gen_ptr + 1) & 0x1F;
            *gen_ptr = if val == 0 { 1 } else { val };
        }
    }

    /// Clear the entire table
    pub fn clear(&self) {
        let buckets = unsafe { &mut *self.buckets.get() };
        for bucket in buckets {
            *bucket = TTBucket::empty();
        }
        unsafe {
            *self.generation.get() = 1;
            *self.used.get() = 0;
        }
    }
}

// ============================================================================
// Re-export old types for API compatibility
// ============================================================================

// TTFlag is already defined above with the same variants (compatible)

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        // Verify our compact structs are the expected sizes
        assert_eq!(
            std::mem::size_of::<TTMove>(),
            40,
            "TTMove should be 40 bytes"
        );
        assert_eq!(
            std::mem::size_of::<TTEntry>(),
            64,
            "TTEntry should be 64 bytes"
        );
        assert_eq!(
            std::mem::size_of::<TTBucket>(),
            64 * ENTRIES_PER_BUCKET,
            "TTBucket should be 256 bytes (4 x 64)"
        );
    }

    #[test]
    fn test_tt_basic_operations() {
        let tt = LocalTranspositionTable::new(1); // 1 MB table

        // Store and probe
        let hash = 0x123456789ABCDEF0u64;
        tt.store(&TTStoreParams {
            hash,
            depth: 5,
            flag: TTFlag::Exact,
            score: 100,
            static_eval: 90,
            is_pv: true,
            best_move: None,
            ply: 0,
        });

        let result = tt.probe(&TTProbeParams {
            hash,
            alpha: -1000,
            beta: 1000,
            depth: 5,
            ply: 0,
            rule50_count: 0,
            rule_limit: 100,
        });
        assert!(result.is_some());
        let (score, eval, _, _) = result.unwrap();
        assert_eq!(score, 100);
        assert_eq!(eval, 90);
    }

    #[test]
    fn test_tt_gen_bound_packing() {
        // Test that generation and bound are packed correctly
        for r#gen in [0u8, 1, 31] {
            for flag in [
                TTFlag::None,
                TTFlag::Exact,
                TTFlag::LowerBound,
                TTFlag::UpperBound,
            ] {
                // Testing with is_pv = true
                let packed_pv = TTEntry::pack_gen_bound(r#gen, true, flag);
                let entry_pv = TTEntry {
                    key: 0,
                    score: 0,
                    eval: 0,
                    depth: 0,
                    gen_bound: packed_pv,
                    _padding: [0; 6],
                    tt_move: TTMove::none(),
                };
                assert_eq!(entry_pv.generation(), r#gen & 0x1F);
                assert_eq!(entry_pv.flag(), flag);
                assert!(entry_pv.is_pv());

                // Testing with is_pv = false
                let packed_nopv = TTEntry::pack_gen_bound(r#gen, false, flag);
                let entry_nopv = TTEntry {
                    key: 0,
                    score: 0,
                    eval: 0,
                    depth: 0,
                    gen_bound: packed_nopv,
                    _padding: [0; 6],
                    tt_move: TTMove::none(),
                };
                assert_eq!(entry_nopv.generation(), r#gen & 0x1F);
                assert_eq!(entry_nopv.flag(), flag);
                assert!(!entry_nopv.is_pv());
            }
        }
    }

    #[test]
    fn test_ttmove_sentinel() {
        let none = TTMove::none();
        assert!(!none.is_some());
        assert!(none.to_move().is_none());

        // A real move should be_some
        let real = TTMove {
            from_x: 0,
            from_y: 0,
            to_x: 1,
            to_y: 1,
            piece_type: 0,
            piece_color: 0,
            promotion: 0,
        };
        assert!(real.is_some());
    }
}
