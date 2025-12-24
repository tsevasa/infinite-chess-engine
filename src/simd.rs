// SIMD Optimization Module for WASM 128-bit SIMD
//
// Provides optimized bitboard operations using WebAssembly SIMD intrinsics.
// Auto-enabled via .cargo/config.toml for WASM targets.
//
// Note: WASM SIMD has limited intrinsics. We use what's available and fall back
// to scalar for operations like popcount that aren't in the stable API.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

/// SIMD-optimized population count for two 64-bit values.
/// Note: WASM SIMD doesn't have native i64x2_popcnt, so we use scalar.
#[inline(always)]
pub fn popcnt_pair(a: u64, b: u64) -> (u32, u32) {
    // Scalar implementation - WASM SIMD doesn't have vector popcount
    (a.count_ones(), b.count_ones())
}

/// SIMD-optimized check if both bitboards are zero (empty).
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn both_zero(a: u64, b: u64) -> bool {
    // Use v128_any_true on the OR of both values
    // If any bit is set, the result is non-zero, so we check !v128_any_true
    let vec = u64x2(a, b);
    !v128_any_true(vec)
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn both_zero(a: u64, b: u64) -> bool {
    a == 0 && b == 0
}

/// SIMD-optimized check if either bitboard is non-zero (has pieces).
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn either_nonzero(a: u64, b: u64) -> bool {
    let vec = u64x2(a, b);
    v128_any_true(vec)
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn either_nonzero(a: u64, b: u64) -> bool {
    a != 0 || b != 0
}

/// SIMD-optimized bitwise OR of two pairs:
/// Returns (a1 | b1, a2 | b2)
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn or_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    let vec_a = u64x2(a1, a2);
    let vec_b = u64x2(b1, b2);
    let result = v128_or(vec_a, vec_b);
    (
        u64x2_extract_lane::<0>(result),
        u64x2_extract_lane::<1>(result),
    )
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn or_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    (a1 | b1, a2 | b2)
}

/// SIMD-optimized bitwise AND of two pairs:
/// Returns (a1 & b1, a2 & b2)
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn and_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    let vec_a = u64x2(a1, a2);
    let vec_b = u64x2(b1, b2);
    let result = v128_and(vec_a, vec_b);
    (
        u64x2_extract_lane::<0>(result),
        u64x2_extract_lane::<1>(result),
    )
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn and_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    (a1 & b1, a2 & b2)
}

/// SIMD-optimized bitwise AND-NOT of two pairs:
/// Returns (a1 & !b1, a2 & !b2)
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn andnot_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    let vec_a = u64x2(a1, a2);
    let vec_b = u64x2(b1, b2);
    let result = v128_andnot(vec_a, vec_b);
    (
        u64x2_extract_lane::<0>(result),
        u64x2_extract_lane::<1>(result),
    )
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn andnot_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    (a1 & !b1, a2 & !b2)
}

/// Sum two i32 accumulators in parallel.
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn add_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (i32, i32) {
    let vec_a = i32x4(a1, a2, 0, 0);
    let vec_b = i32x4(b1, b2, 0, 0);
    let result = i32x4_add(vec_a, vec_b);
    (
        i32x4_extract_lane::<0>(result),
        i32x4_extract_lane::<1>(result),
    )
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn add_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (i32, i32) {
    (a1 + b1, a2 + b2)
}

// ============================================================================
// Higher-Level SIMD Helpers for Chess
// ============================================================================

/// Count pieces for white and black simultaneously in a tile.
#[inline]
pub fn count_pieces_both_colors(occ_white: u64, occ_black: u64) -> (u32, u32) {
    popcnt_pair(occ_white, occ_black)
}

/// Check if a tile has any pieces (either color).
#[inline]
pub fn tile_is_empty(occ_white: u64, occ_black: u64) -> bool {
    both_zero(occ_white, occ_black)
}

/// Check if a tile has pieces of specified color.
#[inline]
pub fn has_pieces_of_either_type(occ_a: u64, occ_b: u64) -> bool {
    either_nonzero(occ_a, occ_b)
}

/// Combine slider bitboards: (bishops | queens, rooks | queens)
#[inline]
pub fn combined_sliders(occ_bishops: u64, occ_rooks: u64, occ_queens: u64) -> (u64, u64) {
    or_pairs(occ_bishops, occ_rooks, occ_queens, occ_queens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcnt_pair() {
        let (a, b) = popcnt_pair(0b1111, 0b11);
        assert_eq!(a, 4);
        assert_eq!(b, 2);

        let (a, b) = popcnt_pair(u64::MAX, 0);
        assert_eq!(a, 64);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_both_zero() {
        assert!(both_zero(0, 0));
        assert!(!both_zero(1, 0));
        assert!(!both_zero(0, 1));
        assert!(!both_zero(1, 1));
    }

    #[test]
    fn test_or_pairs() {
        let (a, b) = or_pairs(0b1100, 0b1010, 0b0011, 0b0101);
        assert_eq!(a, 0b1111);
        assert_eq!(b, 0b1111);
    }

    #[test]
    fn test_and_pairs() {
        let (a, b) = and_pairs(0b1100, 0b1010, 0b1111, 0b0011);
        assert_eq!(a, 0b1100);
        assert_eq!(b, 0b0010);
    }

    #[test]
    fn test_either_nonzero() {
        assert!(!either_nonzero(0, 0));
        assert!(either_nonzero(1, 0));
        assert!(either_nonzero(0, 1));
        assert!(either_nonzero(1, 1));
        assert!(either_nonzero(u64::MAX, u64::MAX));
    }

    #[test]
    fn test_andnot_pairs() {
        // a1 & !b1, a2 & !b2
        let (a, b) = andnot_pairs(0b1111, 0b1111, 0b1100, 0b0011);
        assert_eq!(a, 0b0011); // 1111 & !1100 = 1111 & 0011 = 0011
        assert_eq!(b, 0b1100); // 1111 & !0011 = 1111 & 1100 = 1100

        let (a, b) = andnot_pairs(0xFF, 0xFF, 0xFF, 0xFF);
        assert_eq!(a, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_add_i32_pairs() {
        let (a, b) = add_i32_pairs(10, 20, 5, 15);
        assert_eq!(a, 15);
        assert_eq!(b, 35);

        let (a, b) = add_i32_pairs(-10, -20, 30, 40);
        assert_eq!(a, 20);
        assert_eq!(b, 20);
    }

    #[test]
    fn test_count_pieces_both_colors() {
        let (w, b) = count_pieces_both_colors(0b11111111, 0b1111);
        assert_eq!(w, 8);
        assert_eq!(b, 4);
    }

    #[test]
    fn test_tile_is_empty() {
        assert!(tile_is_empty(0, 0));
        assert!(!tile_is_empty(1, 0));
        assert!(!tile_is_empty(0, 1));
    }

    #[test]
    fn test_has_pieces_of_either_type() {
        assert!(!has_pieces_of_either_type(0, 0));
        assert!(has_pieces_of_either_type(1, 0));
        assert!(has_pieces_of_either_type(0, 1));
    }

    #[test]
    fn test_combined_sliders() {
        // bishops=0b0001, rooks=0b0010, queens=0b0100
        // Result: (bishops|queens, rooks|queens) = (0b0101, 0b0110)
        let (diag, ortho) = combined_sliders(0b0001, 0b0010, 0b0100);
        assert_eq!(diag, 0b0101);
        assert_eq!(ortho, 0b0110);
    }
}
