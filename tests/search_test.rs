use hydrochess_wasm::game::GameState;
use hydrochess_wasm::board::{Piece, PieceType, PlayerColor};

// Helper to setup a specific board state
fn setup_mate_in_one() -> GameState {
    let mut game = GameState::new();
    
    // Scenario: Back-rank mate
    // Black King at (0, 8)
    // Black Pawns at (-1, 7), (0, 7), (1, 7) blocking escape
    // White Rook at (7, 1)
    // White King at (0, 10) blocking escape from other side
    // White to move. Rook (7, 1) -> (7, 8) is mate.
    
    game.board.set_piece(0, 8, Piece::new(PieceType::King, PlayerColor::Black));
    game.board.set_piece(-1, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
    game.board.set_piece(0, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
    game.board.set_piece(1, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
    
    game.board.set_piece(7, 1, Piece::new(PieceType::Rook, PlayerColor::White));
    game.board.set_piece(0, 10, Piece::new(PieceType::King, PlayerColor::White));
    
    game.turn = PlayerColor::White;
    game.material_score = 200; 
    
    game
}

fn setup_2queen_vs_king() -> GameState {
    let mut game = GameState::new();

    game.board.set_piece(0, 8, Piece::new(PieceType::King, PlayerColor::Black));
    game.board.set_piece(3, 9, Piece::new(PieceType::Queen, PlayerColor::White));
    game.board.set_piece(-5, 8, Piece::new(PieceType::Queen, PlayerColor::White));

    game.turn = PlayerColor::White;
    game.material_score = 1800; 

    game
}

#[test]
fn test_mate_in_one() {
    let mut game = setup_mate_in_one();
    
    // Print all legal moves to debug
    let moves = game.get_legal_moves();
    println!("Available moves: {}", moves.len());
    for m in &moves {
        println!("  Move: ({},{}) -> ({},{})", m.from.x, m.from.y, m.to.x, m.to.y);
    }
    
    // Search with iterative deepening (depth 10)
    let best_move = hydrochess_wasm::search::get_best_move(&mut game, 10);
    
    assert!(best_move.is_some(), "Should find a move");
    let m = best_move.unwrap();
    
    println!("Best move chosen: from ({}, {}) to ({}, {})", m.from.x, m.from.y, m.to.x, m.to.y);
    
    // Verify the engine finds the checkmate move: Rook (7,1) -> (7,8)
    assert_eq!(m.from.x, 7, "Move should be from rook at x=7");
    assert_eq!(m.from.y, 1, "Move should be from rook at y=1");
    assert_eq!(m.to.x, 7, "Move should be to x=7");
    assert_eq!(m.to.y, 8, "Move should be to y=8 (checkmate square)");
}


#[test]
fn test_2queen() {
    let mut game = setup_2queen_vs_king();

    // Print all legal moves to debug
    let moves = game.get_legal_moves();
    println!("Available moves: {}", moves.len());
    for m in &moves {
        println!("  Move: ({},{}) -> ({},{})", m.from.x, m.from.y, m.to.x, m.to.y);
    }
    
    // Search with iterative deepening (depth 10)
    let best_move = hydrochess_wasm::search::get_best_move(&mut game, 10);
    
    assert!(best_move.is_some(), "Should find a move");
    let m = best_move.unwrap();
    
    println!("Best move chosen: from ({}, {}) to ({}, {})", m.from.x, m.from.y, m.to.x, m.to.y);
}