use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::moves::{get_legal_moves, Move, is_square_attacked, SpatialIndices};
use crate::evaluation::{get_piece_value, calculate_initial_material};
use std::collections::HashSet;
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct EnPassantState {
    pub square: Coordinate,
    pub pawn_square: Coordinate,
}

#[derive(Clone)]
pub struct UndoMove {
    pub captured_piece: Option<Piece>,
    pub old_en_passant: Option<EnPassantState>,
    pub old_castling_rights: HashSet<Coordinate>,
    pub old_halfmove_clock: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: Board,
    pub turn: PlayerColor,
    pub castling_rights: HashSet<Coordinate>,
    pub en_passant: Option<EnPassantState>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    pub material_score: i32, // Positive = White advantage
    #[serde(skip)]
    pub hash_stack: Vec<u64>, // Position hashes for repetition detection
    #[serde(skip)]
    pub null_moves: u8, // Counter for null moves (for repetition detection)
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            board: Board::new(),
            turn: PlayerColor::White,
            castling_rights: HashSet::new(),
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            material_score: 0,
            hash_stack: Vec::with_capacity(128),
            null_moves: 0,
        }
    }
    
    /// Check for threefold repetition
    pub fn is_threefold(&self) -> bool {
        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }
        
        // Need at least 6 positions to have a potential threefold
        if self.hash_stack.len() < 6 {
            return false;
        }
        
        // Generate current position hash
        let current_hash = self.generate_hash();
        
        let mut repetitions_count = 1;
        // Only look back as far as halfmove_clock allows (captures/pawn moves reset repetition)
        let lookback = (self.halfmove_clock as usize).min(self.hash_stack.len());
        let from = self.hash_stack.len().saturating_sub(lookback);
        let to = self.hash_stack.len().saturating_sub(1);
        
        if to <= from {
            return false;
        }
        
        // Check every other position (same side to move)
        for hash_index in (from..to).rev().step_by(2) {
            if self.hash_stack[hash_index] == current_hash {
                repetitions_count += 1;
                
                if repetitions_count >= 3 {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Check if this is a lone king endgame (one side only has a king)
    pub fn is_lone_king_endgame(&self) -> bool {
        use crate::board::{PieceType, PlayerColor};
        
        let mut white_has_non_king = false;
        let mut black_has_non_king = false;
        
        for (_, piece) in &self.board.pieces {
            if piece.piece_type != PieceType::King {
                if piece.color == PlayerColor::White {
                    white_has_non_king = true;
                } else {
                    black_has_non_king = true;
                }
            }
        }
        
        // One side has only a king (or nothing)
        !white_has_non_king || !black_has_non_king
    }
    
    /// Check if position is a draw by 50-move rule
    pub fn is_fifty(&self) -> bool {
        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }
        self.halfmove_clock >= 100
    }
    
    /// Make a null move (just flip turn, for null move pruning)
    pub fn make_null_move(&mut self) {
        // Push current hash
        let current_hash = self.generate_hash();
        self.hash_stack.push(current_hash);
        
        // Clear en passant
        self.en_passant = None;
        
        // Flip turn
        self.turn = match self.turn {
            PlayerColor::White => PlayerColor::Black,
            PlayerColor::Black => PlayerColor::White,
        };
        
        self.null_moves += 1;
    }
    
    /// Unmake a null move
    pub fn unmake_null_move(&mut self) {
        // Pop hash
        self.hash_stack.pop();
        
        // Flip turn back
        self.turn = match self.turn {
            PlayerColor::White => PlayerColor::Black,
            PlayerColor::Black => PlayerColor::White,
        };
        
        self.null_moves -= 1;
    }
    
    /// Generate a hash for the current position
    pub fn generate_hash(&self) -> u64 {
        use crate::search::TranspositionTable;
        TranspositionTable::generate_hash(self)
    }



    /// Returns pseudo-legal moves. Legality (not leaving king in check) 
    /// is checked in the search after making each move.
    pub fn get_legal_moves(&self) -> Vec<Move> {
        get_legal_moves(&self.board, self.turn, &self.castling_rights, &self.en_passant)
    }
    
    /// Check if the side that just moved left their king in check (illegal move).
    /// Call this AFTER make_move to verify legality.
    pub fn is_move_illegal(&self) -> bool {
        // After make_move, self.turn is the opponent.
        // We need to check if the side that just moved (opponent of current turn) is in check.
        let moved_color = match self.turn {
            PlayerColor::White => PlayerColor::Black,
            PlayerColor::Black => PlayerColor::White,
        };
        
        // Find king of the side that just moved
        let king_pos = self.board.pieces.iter()
            .find(|(_, p)| p.piece_type == PieceType::King && p.color == moved_color)
            .map(|((x, y), _)| Coordinate::new(*x, *y));

        if let Some(king_pos) = king_pos {
            let indices = SpatialIndices::new(&self.board);
            // Check if the opponent (current turn) attacks the king
            is_square_attacked(&self.board, &king_pos, self.turn, Some(&indices))
        } else {
            false // No king found - shouldn't happen
        }
    }

    pub fn is_in_check(&self) -> bool {
        // Find king
        let king_pos = self.board.pieces.iter()
            .find(|(_, p)| p.piece_type == PieceType::King && p.color == self.turn)
            .map(|((x, y), _)| Coordinate::new(*x, *y));

        if let Some(king_pos) = king_pos {
            let indices = SpatialIndices::new(&self.board);
            let attacker_color = match self.turn {
                PlayerColor::White => PlayerColor::Black,
                PlayerColor::Black => PlayerColor::White,
            };
            is_square_attacked(&self.board, &king_pos, attacker_color, Some(&indices))
        } else {
            false // No king? Not in check (or invalid state)
        }
    }

    pub fn make_move(&mut self, m: &Move) -> UndoMove {
        // Push current position hash BEFORE making the move (for repetition detection)
        let current_hash = self.generate_hash();
        self.hash_stack.push(current_hash);
        
        let piece = self.board.remove_piece(&m.from.x, &m.from.y).unwrap();
        
        let undo_info = UndoMove {
            captured_piece: self.board.get_piece(&m.to.x, &m.to.y).cloned(),
            old_en_passant: self.en_passant.clone(),
            old_castling_rights: self.castling_rights.clone(),
            old_halfmove_clock: self.halfmove_clock,
        };

        // Handle captures (reset halfmove clock)
        let is_capture = undo_info.captured_piece.is_some();
        
        if let Some(captured) = &undo_info.captured_piece {
            let value = get_piece_value(captured.piece_type);
            if captured.color == PlayerColor::White {
                self.material_score -= value;
            } else {
                self.material_score += value;
            }
        }
        
        // Handle En Passant capture
        let mut is_ep_capture = false;
        if piece.piece_type == PieceType::Pawn {
            if let Some(ep) = &self.en_passant {
                if m.to.x == ep.square.x && m.to.y == ep.square.y {
                    if let Some(captured_pawn) = self.board.remove_piece(&ep.pawn_square.x, &ep.pawn_square.y) {
                        is_ep_capture = true;
                        // Update material for EP capture
                        let value = get_piece_value(captured_pawn.piece_type);
                        if captured_pawn.color == PlayerColor::White {
                            self.material_score -= value;
                        } else {
                            self.material_score += value;
                        }
                    }
                }
            }
        }

        // Handle Promotion
        if let Some(promo_str) = &m.promotion {
             // Remove pawn value
             let pawn_val = get_piece_value(PieceType::Pawn);
             if piece.color == PlayerColor::White {
                 self.material_score -= pawn_val;
             } else {
                 self.material_score += pawn_val;
             }
             
             // Add promoted piece value
             let promo_type = match promo_str.as_str() {
                 "q" => PieceType::Queen,
                 "r" => PieceType::Rook,
                 "b" => PieceType::Bishop,
                 "n" => PieceType::Knight,
                 _ => PieceType::Queen,
             };
             
             let promo_val = get_piece_value(promo_type);
             if piece.color == PlayerColor::White {
                 self.material_score += promo_val;
             } else {
                 self.material_score -= promo_val;
             }
        }

        // Update castling rights
        if piece.piece_type == PieceType::King {
            self.castling_rights.remove(&m.from);
        } else if piece.piece_type == PieceType::Rook {
            self.castling_rights.remove(&m.from);
        }
        // If rook is captured
        if is_capture {
             self.castling_rights.remove(&m.to);
        }

        // Handle Castling Move (King moves > 1 square)
        if piece.piece_type == PieceType::King {
            let dx = m.to.x - m.from.x;
            if dx.abs() > 1 {
                // Castling!
                if let Some(rook_coord) = &m.rook_coord {
                     if let Some(rook) = self.board.remove_piece(&rook_coord.x, &rook_coord.y) {
                        let rook_to_x = m.from.x + (if dx > 0 { 1 } else { -1 });
                        self.board.set_piece(rook_to_x, m.from.y, rook);
                    }
                }
            }
        }

        // Move piece (handle promotion if needed)
        let final_piece = if let Some(promo_str) = &m.promotion {
             let promo_type = match promo_str.as_str() {
                 "q" => PieceType::Queen,
                 "r" => PieceType::Rook,
                 "b" => PieceType::Bishop,
                 "n" => PieceType::Knight,
                 _ => PieceType::Queen,
             };
             Piece::new(promo_type, piece.color)
        } else {
            piece.clone()
        };

        self.board.set_piece(m.to.x, m.to.y, final_piece);

        // Update En Passant state
        self.en_passant = None;
        if piece.piece_type == PieceType::Pawn {
            let dy = m.to.y - m.from.y;
            if dy.abs() == 2 {
                let ep_y = m.from.y + (dy / 2);
                self.en_passant = Some(EnPassantState {
                    square: Coordinate::new(m.from.x, ep_y),
                    pawn_square: m.to.clone(),
                });
            }
        }

        // Update clocks
        if piece.piece_type == PieceType::Pawn || is_capture || is_ep_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        if self.turn == PlayerColor::Black {
            self.fullmove_number += 1;
        }

        self.turn = match self.turn {
            PlayerColor::White => PlayerColor::Black,
            PlayerColor::Black => PlayerColor::White,
        };
        
        undo_info
    }

    pub fn undo_move(&mut self, m: &Move, undo: UndoMove) {
        // Pop the hash that was pushed in make_move
        self.hash_stack.pop();
        
        // Revert turn
        self.turn = match self.turn {
            PlayerColor::White => PlayerColor::Black,
            PlayerColor::Black => PlayerColor::White,
        };
        
        if self.turn == PlayerColor::Black {
            self.fullmove_number -= 1;
        }

        // Revert piece move
        // Get the piece from the 'to' square
        let mut piece = self.board.remove_piece(&m.to.x, &m.to.y).unwrap();
        
        // Handle Promotion Revert
        if m.promotion.is_some() {
            // Convert back to pawn
            let promo_val = get_piece_value(piece.piece_type);
            let pawn_val = get_piece_value(PieceType::Pawn);
            
            if piece.color == PlayerColor::White {
                self.material_score -= promo_val;
                self.material_score += pawn_val;
            } else {
                self.material_score += promo_val;
                self.material_score -= pawn_val;
            }
            piece.piece_type = PieceType::Pawn;
        }

        // Move back to 'from'
        self.board.set_piece(m.from.x, m.from.y, piece.clone());

        // Restore captured piece
        if let Some(captured) = undo.captured_piece {
            let value = get_piece_value(captured.piece_type);
            if captured.color == PlayerColor::White {
                self.material_score += value;
            } else {
                self.material_score -= value;
            }
            self.board.set_piece(m.to.x, m.to.y, captured);
        }

        // Handle En Passant Capture Revert
        // If it was an EP capture, the captured pawn was on 'pawn_square' of the OLD en_passant state
        // But wait, we don't store "is_ep_capture" in UndoMove.
        // We can infer it: if piece is pawn, and to_square matches old_ep.square
        if piece.piece_type == PieceType::Pawn {
             if let Some(ep) = &undo.old_en_passant {
                 if m.to.x == ep.square.x && m.to.y == ep.square.y {
                     // It was an EP capture!
                     // Restore the captured pawn
                     let captured_pawn = Piece::new(PieceType::Pawn, match piece.color {
                         PlayerColor::White => PlayerColor::Black,
                         PlayerColor::Black => PlayerColor::White,
                     });
                     
                     self.board.set_piece(ep.pawn_square.x, ep.pawn_square.y, captured_pawn.clone());
                     
                     // Restore material
                     let value = get_piece_value(PieceType::Pawn);
                     if captured_pawn.color == PlayerColor::White {
                         self.material_score += value;
                     } else {
                         self.material_score -= value;
                     }
                 }
             }
        }

        // Handle Castling Revert
        if piece.piece_type == PieceType::King {
            let dx = m.to.x - m.from.x;
            if dx.abs() > 1 {
                // Castling was performed. Move rook back.
                if let Some(rook_coord) = &m.rook_coord {
                    let rook_to_x = m.from.x + (if dx > 0 { 1 } else { -1 });
                    if let Some(rook) = self.board.remove_piece(&rook_to_x, &m.from.y) {
                        self.board.set_piece(rook_coord.x, rook_coord.y, rook);
                    }
                }
            }
        }

        // Restore state
        self.en_passant = undo.old_en_passant;
        self.castling_rights = undo.old_castling_rights;
        self.halfmove_clock = undo.old_halfmove_clock;
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = self.get_legal_moves();
        let mut nodes = 0;

        for m in moves {
            let undo = self.make_move(&m);
            nodes += self.perft(depth - 1);
            self.undo_move(&m, undo);
        }

        nodes
    }

    pub fn setup_standard_chess(&mut self) {
        self.board = Board::new();
        self.castling_rights.clear();
        self.en_passant = None;
        self.turn = PlayerColor::White;
        self.halfmove_clock = 0;
        self.fullmove_number = 1;
        self.material_score = 0;

        // White Pieces
        self.board.set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        self.board.set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        self.board.set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        self.board.set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        self.board.set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        self.board.set_piece(6, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        self.board.set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        self.board.set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White));

        for x in 1..=8 {
            self.board.set_piece(x, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        }

        // Black Pieces
        self.board.set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        self.board.set_piece(2, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        self.board.set_piece(3, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        self.board.set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
        self.board.set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        self.board.set_piece(6, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        self.board.set_piece(7, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        self.board.set_piece(8, 8, Piece::new(PieceType::Rook, PlayerColor::Black));

        for x in 1..=8 {
            self.board.set_piece(x, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
        }

        // Castling Rights
        self.castling_rights.insert(Coordinate::new(1, 1));
        self.castling_rights.insert(Coordinate::new(5, 1));
        self.castling_rights.insert(Coordinate::new(8, 1));
        
        self.castling_rights.insert(Coordinate::new(1, 8));
        self.castling_rights.insert(Coordinate::new(5, 8));
        self.castling_rights.insert(Coordinate::new(8, 8));
        
        // Calculate initial material
        self.material_score = calculate_initial_material(&self.board);
    }
}
