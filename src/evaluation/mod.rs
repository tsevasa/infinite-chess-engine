// Modular Evaluation System
//
// Design Principles:
// 1. `base` contains ALL default heuristics
// 2. Variant files in `variants/` ONLY exist if they have special logic

pub mod base;
pub mod helpers;
pub mod insufficient_material;
pub mod mop_up;
pub mod pieces;
pub mod variants;

use crate::Variant;
use crate::board::PlayerColor;
use crate::game::GameState;

// Re-export commonly used items
pub use base::{calculate_initial_material, evaluate_lazy, get_piece_value};

#[cfg(feature = "eval_tuning")]
pub use base::{EvalFeatures, reset_eval_features, snapshot_eval_features};

/// Main evaluation entry point.
#[inline]
pub fn evaluate(game: &GameState) -> i32 {
    match game.variant {
        Some(Variant::Chess) => variants::chess::evaluate(game),
        Some(Variant::ConfinedClassical) => variants::confined_classical::evaluate(game),
        Some(Variant::Obstocean) => variants::obstocean::evaluate(game),
        Some(Variant::PawnHorde) => variants::pawn_horde::evaluate(game),
        Some(Variant::Palace) => variants::palace::evaluate(game),
        // Add new variants here as they get custom evaluators
        _ => base::evaluate(game), // Default: use base for all others
    }
}

/// Fast evaluation for use - just material + basic positional
#[allow(dead_code)]
#[inline]
pub fn evaluate_fast(game: &GameState) -> i32 {
    let score = game.material_score;

    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece, PieceType, PlayerColor};
    use crate::game::GameState;

    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();
        game
    }

    #[test]
    fn test_evaluate_returns_value() {
        let game = create_test_game();
        let score = evaluate(&game);
        // K vs K should be close to 0
        assert!(score.abs() < 1000, "K vs K should be near 0");
    }

    #[test]
    fn test_evaluate_material_advantage() {
        let mut game = GameState::new();
        game.board = Board::new();

        // More complex position to avoid lone king detection
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();

        // Recalculate material score
        let mut score = 0i32;
        for (_, piece) in game.board.iter() {
            let val = get_piece_value(piece.piece_type());
            match piece.color() {
                PlayerColor::White => score += val,
                PlayerColor::Black => score -= val,
                PlayerColor::Neutral => {}
            }
        }
        game.material_score = score;

        let eval = evaluate(&game);
        // Just verify we get a reasonable value (queen > rook typically)
        // Exact values depend on evaluation logic
        assert!(
            eval.abs() < 100000,
            "Eval should be in reasonable range, got {}",
            eval
        );
    }

    #[test]
    fn test_evaluate_fast_white_turn() {
        let mut game = create_test_game();
        game.material_score = 500;
        game.turn = PlayerColor::White;

        let score = evaluate_fast(&game);
        assert_eq!(score, 500);
    }

    #[test]
    fn test_evaluate_fast_black_turn() {
        let mut game = create_test_game();
        game.material_score = 500;
        game.turn = PlayerColor::Black;

        let score = evaluate_fast(&game);
        assert_eq!(score, -500, "Should negate for black's perspective");
    }

    #[test]
    fn test_evaluate_with_variant() {
        let mut game = create_test_game();
        game.variant = Some(crate::Variant::Chess);

        // Should dispatch to Chess evaluator
        let _score = evaluate(&game);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_get_piece_value() {
        // Test piece values are reasonable
        let queen_val = get_piece_value(PieceType::Queen);
        let rook_val = get_piece_value(PieceType::Rook);
        let bishop_val = get_piece_value(PieceType::Bishop);
        let knight_val = get_piece_value(PieceType::Knight);
        let pawn_val = get_piece_value(PieceType::Pawn);

        assert!(queen_val > rook_val, "Queen should be worth more than rook");
        assert!(
            rook_val > bishop_val,
            "Rook should be worth more than bishop"
        );
        assert!(bishop_val >= knight_val, "Bishop should be worth >= knight");
        assert!(
            knight_val > pawn_val,
            "Knight should be worth more than pawn"
        );
        assert!(pawn_val > 0, "Pawn should have positive value");
    }
}
