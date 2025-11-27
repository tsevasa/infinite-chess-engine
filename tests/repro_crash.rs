use hydrochess_wasm::game::GameState;
use hydrochess_wasm::board::{Board, Piece, PieceType, PlayerColor, Coordinate};
use std::collections::HashSet;

#[test]
fn test_repro_crash() {
    let mut board = Board::new();
    
    // Setup pieces as per the log
    // White Pawns at y=2, x=1..8
    for x in 1..=8 {
        board.set_piece(x, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
    }
    // Black Pawns at y=7, x=1..8
    for x in 1..=8 {
        board.set_piece(x, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
    }
    
    // Rooks
    board.set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
    board.set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White));
    board.set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
    board.set_piece(8, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
    
    // Knights
    board.set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(2, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
    board.set_piece(7, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
    
    // Bishops
    board.set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
    board.set_piece(6, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
    board.set_piece(3, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
    board.set_piece(6, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
    
    // Queens
    board.set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
    board.set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
    
    // Kings
    board.set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));

    // Castling rights (includes pawns as per log)
    let mut castling_rights = HashSet::new();
    // White Pawns
    for x in 1..=8 { castling_rights.insert(Coordinate::new(x, 2)); }
    // Black Pawns
    for x in 1..=8 { castling_rights.insert(Coordinate::new(x, 7)); }
    // Rooks and Kings
    castling_rights.insert(Coordinate::new(1, 1));
    castling_rights.insert(Coordinate::new(8, 1));
    castling_rights.insert(Coordinate::new(5, 1));
    
    castling_rights.insert(Coordinate::new(1, 8));
    castling_rights.insert(Coordinate::new(8, 8));
    castling_rights.insert(Coordinate::new(5, 8));

    let mut game = GameState {
        board,
        turn: PlayerColor::White,
        castling_rights,
        en_passant: None,
        halfmove_clock: 0,
        fullmove_number: 1,
        material_score: 0,
        hash_stack: Vec::new(),
        null_moves: 0,
    };

    println!("Starting search...");
    // Updated signature: get_best_move(&mut GameState, max_depth)
    let best_move = hydrochess_wasm::search::get_best_move(&mut game, 4); // Depth 4 to trigger some search
    println!("Best move: {:?}", best_move);
}
