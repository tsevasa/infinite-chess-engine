//! Zobrist hashing for infinite chess.
//! 
//! Uses computed hashes based on coordinates since we can't pre-compute a table
//! for an infinite board. The hash is maintained incrementally in GameState.

use crate::board::{PieceType, PlayerColor, Coordinate};

/// Number of piece types (used for indexing into piece keys)
const NUM_PIECE_TYPES: usize = 22;
const NUM_COLORS: usize = 3; // White, Black, Neutral

/// Pre-computed random keys for piece-type combinations (not position-dependent)
/// We mix these with position hashes at runtime
static PIECE_KEYS: [[u64; NUM_COLORS]; NUM_PIECE_TYPES] = {
    // Use a simple PRNG to generate constants at compile time
    const fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    
    let mut keys = [[0u64; NUM_COLORS]; NUM_PIECE_TYPES];
    let mut seed = 0x123456789ABCDEF0u64;
    
    let mut i = 0;
    while i < NUM_PIECE_TYPES {
        let mut j = 0;
        while j < NUM_COLORS {
            seed = splitmix64(seed);
            keys[i][j] = seed;
            j += 1;
        }
        i += 1;
    }
    keys
};

/// Key for side to move (XOR when black to move)
pub const SIDE_KEY: u64 = 0x9E3779B97F4A7C15;

/// Keys for castling rights (indexed by normalized coordinate hash)
const CASTLING_KEY_MIXER: u64 = 0xDEADBEEF12345678;

/// Key for en passant file
const EN_PASSANT_KEY_MIXER: u64 = 0xCAFEBABE87654321;

/// Normalize coordinate for hashing (handle infinite board via bucketing)
///
/// This mirrors the old TT behaviour: coordinates within [-BOUND, BOUND]
/// are kept distinct; far-away squares are wrapped into BUCKETS-sized
/// buckets at the edges, preserving some translation invariance.
#[inline(always)]
fn normalize_coord(coord: i64) -> i32 {
    const BOUND: i64 = 150;
    const BUCKETS: i64 = 8;

    if coord.abs() <= BOUND {
        coord as i32
    } else {
        let sign = coord.signum();
        let delta = (coord - sign * BOUND) % BUCKETS;
        (sign * BOUND + delta) as i32
    }
}

/// Hash a coordinate into a u64
/// Uses a fast mixing function on *bucketed* coordinates, preserving
/// the infinite-board semantics while being efficient for incremental use.
#[inline(always)]
pub fn hash_coordinate(x: i64, y: i64) -> u64 {
    let nx = normalize_coord(x) as u64;
    let ny = normalize_coord(y) as u64;

    // Fast mixing - fewer operations, good enough distribution
    let h = nx.wrapping_mul(0x517cc1b727220a95)
        ^ ny.wrapping_mul(0x9e3779b97f4a7c15);
    h ^ (h >> 32)
}

/// Get the Zobrist key for a piece at a position
#[inline(always)]
pub fn piece_key(piece_type: PieceType, color: PlayerColor, x: i64, y: i64) -> u64 {
    let coord_hash = hash_coordinate(x, y);
    let pk = PIECE_KEYS[piece_type as usize][color as usize];
    coord_hash ^ pk
}

/// Get the key for a special right (castling/pawn double-move) at a coordinate
#[inline(always)]
pub fn special_right_key(coord: &Coordinate) -> u64 {
    hash_coordinate(coord.x, coord.y) ^ CASTLING_KEY_MIXER
}

/// Get the key for en passant square
#[inline(always)]
pub fn en_passant_key(x: i64, y: i64) -> u64 {
    hash_coordinate(x, y) ^ EN_PASSANT_KEY_MIXER
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_piece_keys_unique() {
        // Verify piece keys are reasonably unique
        let mut keys = Vec::new();
        for i in 0..NUM_PIECE_TYPES {
            for j in 0..NUM_COLORS {
                keys.push(PIECE_KEYS[i][j]);
            }
        }
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), NUM_PIECE_TYPES * NUM_COLORS);
    }
    
    #[test]
    fn test_coordinate_hash_different() {
        let h1 = hash_coordinate(1, 1);
        let h2 = hash_coordinate(1, 2);
        let h3 = hash_coordinate(2, 1);
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }
}
