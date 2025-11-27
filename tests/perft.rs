use hydrochess_wasm::game::GameState;
use hydrochess_wasm::board::{Coordinate, Piece, PieceType, PlayerColor};
use hydrochess_wasm::moves::{Move, get_legal_moves};
use std::time::Instant;
use std::collections::HashSet;

#[test]
fn run_perft_suite() {
    println!("\n================================================================");
    println!("Running Perft Suite for HydroChess WASM v2 (i64 Optimized)");
    println!("================================================================");
    println!("NOTE: Move counts vary based on slider move heuristics");
    println!("================================================================");

    // Setup Standard Chess
    let mut game = GameState::new();
    game.setup_standard_chess();

    println!("Initial Board Setup: Standard Chess");
    println!("----------------------------------------------------------------");

    // Depth 1
    let start = Instant::now();
    let nodes_d1 = game.perft(1);
    let duration_d1 = start.elapsed();
    println!("Depth 1: {:10} nodes | Time: {:?} | NPS: {:10}", 
        nodes_d1, 
        duration_d1, 
        (nodes_d1 as u128 * 1_000_000) / duration_d1.as_micros().max(1)
    );

    // Depth 2
    let start = Instant::now();
    let nodes_d2 = game.perft(2);
    let duration_d2 = start.elapsed();
    println!("Depth 2: {:10} nodes | Time: {:?} | NPS: {:10}", 
        nodes_d2, 
        duration_d2,
        (nodes_d2 as u128 * 1_000_000) / duration_d2.as_micros().max(1)
    );

    // Depth 3
    let start = Instant::now();
    let nodes_d3 = game.perft(3);
    let duration_d3 = start.elapsed();
    println!("Depth 3: {:10} nodes | Time: {:?} | NPS: {:10}", 
        nodes_d3, 
        duration_d3,
        (nodes_d3 as u128 * 1_000_000) / duration_d3.as_micros().max(1)
    );

    println!("================================================================");
    println!("Performance Summary:");
    println!("  Depth 3 completed in {:?}", duration_d3);
    if duration_d3.as_secs_f64() < 1.0 {
        println!("  ✓ Target achieved: < 1 second for Depth 3");
    } else {
        println!("  ✗ Target not met: Depth 3 took > 1 second");
    }
    println!("================================================================");
}

#[test]
fn replay_bug_move_sequence() {
    println!("\n================================================================");
    println!("Replaying suspected bug move sequence");
    println!("================================================================");

    let mut game = GameState::new();
    game.setup_standard_chess();

    // First: make a simple white opening move so that the following
    // sequence starts with Black to move, matching the JS logs.
    {
        let fx = 1;
        let fy = 2;
        let tx = 1;
        let ty = 3;
        let piece = game
            .board
            .get_piece(&fx, &fy)
            .cloned()
            .expect("Expected white pawn at (1,2) for opening move");
        let mv = Move::new(Coordinate::new(fx, fy), Coordinate::new(tx, ty), piece);
        let _undo = game.make_move(&mv);
        assert!(
            !game.is_move_illegal(),
            "Opening white move (1,2)->(1,3) is illegal"
        );
    }

    // Sequence captured from JS engine logs (in board coordinates)
    // This now correctly starts with Black to move after White's pawn move.
    let moves: &[(i64, i64, i64, i64)] = &[
        // Move 1 - black
        (1, 8, 0, 8),
        // Move 2 - white
        (8, 1, 9, 1),
        // Move 3 - black
        (8, 8, 10, 8),
        // Move 4 - white
        (1, 1, -1, 1),
        // Move 5 - black
        (7, 8, 6, 6),
        // Move 6 - white
        (2, 1, 3, 3),
        // Move 7 - black
        (2, 8, 3, 6),
        // Move 8 - white
        (-1, 1, -1, 8),
        // Move 9 - black
        (0, 8, -1, 8),
        // Move 10 - white
        (6, 1, -1, 8),
        // Move 11 - black
        (10, 8, 10, 2),
        // Move 12 - white
        (-1, 8, 2, 11),
    ];

    for (idx, (fx, fy, tx, ty)) in moves.iter().cloned().enumerate() {
        println!(
            "Applying move {}: from ({}, {}) to ({}, {})",
            idx + 1,
            fx,
            fy,
            tx,
            ty
        );

        // Before move 10 (idx=9), check if the bishop move is actually legal
        if idx == 9 {
            println!("\n>>> Checking legal moves BEFORE move 10 <<<");
            let legal_moves = get_legal_moves(&game.board, game.turn, &game.castling_rights, &game.en_passant);
            
            // Find bishop moves from (6,1)
            let bishop_moves: Vec<_> = legal_moves
                .iter()
                .filter(|m| m.from.x == 6 && m.from.y == 1)
                .collect();
            
            println!("Pieces at (6,1): {:?}", game.board.get_piece(&6, &1));
            println!("Pieces at (5,2): {:?}", game.board.get_piece(&5, &2));
            println!("Bishop at (6,1) has {} legal moves:", bishop_moves.len());
            for m in &bishop_moves {
                println!("  -> ({}, {})", m.to.x, m.to.y);
            }
            
            // Check if the illegal move (-1,8) is in the list
            let has_illegal = bishop_moves.iter().any(|m| m.to.x == -1 && m.to.y == 8);
            if has_illegal {
                println!("!!! BUG: Move to (-1,8) is in legal moves but should be blocked!");
            } else {
                println!("OK: Move to (-1,8) is NOT in legal moves (correctly blocked)");
            }
            println!("<<<\n");
        }

        // Fetch the moving piece from the current board position
        let piece = match game.board.get_piece(&fx, &fy) {
            Some(p) => p.clone(),
            None => {
                panic!(
                    "No piece found at from-square ({}, {}) before move {}",
                    fx,
                    fy,
                    idx + 1
                );
            }
        };

        let mv = Move::new(
            Coordinate::new(fx, fy),
            Coordinate::new(tx, ty),
            piece,
        );

        // Apply move directly through GameState
        let _undo = game.make_move(&mv);

        // If this triggers a panic or illegal state, the test will clearly surface it
        if game.is_move_illegal() {
            panic!("Move {} left side to move in check (illegal).", idx + 1);
        }
    }

    println!("Finished replaying move sequence without immediate illegal-move detection.");
    println!("================================================================");
}

#[test]
fn test_bishop_blocked_by_pawn() {
    println!("\n================================================================");
    println!("Testing: Bishop should NOT be able to move through a pawn");
    println!("================================================================");

    // Setup: standard position, white bishop at (6,1), white pawn at (5,2)
    // The bishop should NOT be able to reach (-1,8) because the pawn blocks the diagonal
    let mut game = GameState::new();
    game.setup_standard_chess();

    // Get all moves for white
    let moves = get_legal_moves(&game.board, PlayerColor::White, &game.castling_rights, &game.en_passant);

    // Find all bishop moves from (6,1)
    let bishop_moves: Vec<_> = moves
        .iter()
        .filter(|m| m.from.x == 6 && m.from.y == 1 && m.piece.piece_type == PieceType::Bishop)
        .collect();

    println!("Bishop at (6,1) has {} moves:", bishop_moves.len());
    for m in &bishop_moves {
        println!("  -> ({}, {})", m.to.x, m.to.y);
    }

    // The bishop at (6,1) moving up-left is blocked by pawn at (5,2)
    // So it should NOT be able to reach any of: (4,3), (3,4), (2,5), (1,6), (0,7), (-1,8)
    let blocked_squares = [
        (4, 3), (3, 4), (2, 5), (1, 6), (0, 7), (-1, 8)
    ];

    for (bx, by) in blocked_squares {
        let illegal_move = bishop_moves.iter().any(|m| m.to.x == bx && m.to.y == by);
        if illegal_move {
            panic!(
                "BUG: Bishop at (6,1) can move to ({}, {}) but pawn at (5,2) should block it!",
                bx, by
            );
        }
    }

    // Also check that it CAN reach squares in other directions (down-right, down-left, up-right)
    // Up-right (toward h3): blocked by pawn at (7,2)
    // Down-left: nothing there in standard position
    // Down-right: nothing there

    println!("✓ Bishop correctly blocked by pawn at (5,2)");
    println!("================================================================");
}

#[test]
fn test_rook_blocked_by_piece() {
    println!("\n================================================================");
    println!("Testing: Rook should NOT be able to move through other pieces");
    println!("================================================================");

    let mut game = GameState::new();
    game.setup_standard_chess();

    // Get all moves for white
    let moves = get_legal_moves(&game.board, PlayerColor::White, &game.castling_rights, &game.en_passant);

    // Find all rook moves from (1,1)
    let rook_moves: Vec<_> = moves
        .iter()
        .filter(|m| m.from.x == 1 && m.from.y == 1 && m.piece.piece_type == PieceType::Rook)
        .collect();

    println!("Rook at (1,1) has {} moves:", rook_moves.len());
    for m in &rook_moves {
        println!("  -> ({}, {})", m.to.x, m.to.y);
    }

    // In standard position, rook at (1,1) is completely blocked:
    // - Up: pawn at (1,2)
    // - Right: knight at (2,1)
    // So it should have NO moves going up past (1,2) or right past (2,1)

    // Check that rook can't reach squares beyond blockers
    let blocked_up = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)];
    let blocked_right = [(3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)];

    for (bx, by) in blocked_up.iter().chain(blocked_right.iter()) {
        let illegal_move = rook_moves.iter().any(|m| m.to.x == *bx && m.to.y == *by);
        if illegal_move {
            panic!(
                "BUG: Rook at (1,1) can move to ({}, {}) but should be blocked!",
                bx, by
            );
        }
    }

    println!("✓ Rook correctly blocked by pieces");
    println!("================================================================");
}

#[test]
fn replay_stuck_sequence_17_moves() {
    println!("\n================================================================");
    println!("Replaying 17-move sequence from SPRT JS logs");
    println!("================================================================");

    let mut game = GameState::new();
    game.setup_standard_chess();

    // Moves extracted from game_log.txt (board coordinates):
    let moves: &[(i64, i64, i64, i64)] = &[
        (5, 1, 5, 0),  // 1. White king
        (8, 8, 9, 8),  // 2. Black rook
        (5, 0, 5, 1),  // 3. White king back
        (1, 8, 0, 8),  // 4. Black rook
        (1, 1, -1, 1), // 5. White rook
        (2, 8, 3, 6),  // 6. Black knight
        (8, 1, 10, 1), // 7. White rook
        (3, 6, 2, 4),  // 8. Black knight
        (2, 1, 3, 3),  // 9. White knight
        (7, 8, 6, 6),  // 10. Black knight
        (1, 2, 1, 3),  // 11. White pawn
        (2, 4, 4, 5),  // 12. Black knight
        (7, 1, 6, 3),  // 13. White knight
        (4, 5, 3, 3),  // 14. Black knight capture
        (2, 2, 3, 3),  // 15. White pawn capture
        (6, 6, 5, 4),  // 16. Black knight
        (6, 3, 5, 5),  // 17. White knight
    ];

    for (idx, (fx, fy, tx, ty)) in moves.iter().cloned().enumerate() {
        println!(
            "Applying move {}: from ({}, {}) to ({}, {})",
            idx + 1, fx, fy, tx, ty
        );

        let piece = match game.board.get_piece(&fx, &fy) {
            Some(p) => p.clone(),
            None => {
                panic!(
                    "No piece found at from-square ({}, {}) before move {}",
                    fx, fy, idx + 1
                );
            }
        };

        let mv = Move::new(Coordinate::new(fx, fy), Coordinate::new(tx, ty), piece);
        let _undo = game.make_move(&mv);

        if game.is_move_illegal() {
            panic!("Move {} leaves side to move in check (illegal)", idx + 1);
        }
    }

    println!("Finished 17-move sequence without illegal-move detection.");
    
    // Now try to get the best move for Black from this position
    // This is what hangs in JS - let's see if it hangs in pure Rust too
    println!("\n>>> Attempting search from this position (Black to move) <<<");
    println!("Turn: {:?}", game.turn);
    
    let legal_moves = get_legal_moves(&game.board, game.turn, &game.castling_rights, &game.en_passant);
    println!("Legal moves for {:?}: {}", game.turn, legal_moves.len());
    
    // Try a timed search with 100ms limit
    use hydrochess_wasm::search;
    let start = std::time::Instant::now();
    let best_move = search::get_best_move_timed(&mut game, 50, 100, true);
    let elapsed = start.elapsed();
    
    println!("Search completed in {:?}", elapsed);
    match best_move {
        Some(m) => println!("Best move: ({},{}) -> ({},{})", m.from.x, m.from.y, m.to.x, m.to.y),
        None => println!("No move found!"),
    }
    
    println!("================================================================");
}
