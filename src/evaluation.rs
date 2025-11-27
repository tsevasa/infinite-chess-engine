use crate::board::{PieceType, PlayerColor, Board, Coordinate};
use crate::game::GameState;

// ==================== Piece Values ====================

pub fn get_piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        // orthodox
        PieceType::Pawn => 100,
        PieceType::Knight => 280,
        PieceType::Bishop => 340,
        PieceType::Rook => 620,
        PieceType::Queen | PieceType::RoyalQueen => 1100,
        PieceType::King | PieceType::Guard => 280,

        // short / medium range
        PieceType::Camel => 270,   // (1,3) leaper
        PieceType::Giraffe => 260, // (1,4) leaper
        PieceType::Zebra => 260,   // (2,3) leaper

        // riders / compounds
        PieceType::Knightrider => 700,
        PieceType::Amazon => 1400,
        PieceType::Hawk => 600,
        PieceType::Chancellor => 1000,
        PieceType::Archbishop => 900,
        PieceType::Centaur => 550,

        // royal compound
        PieceType::RoyalCentaur => 620,

        // special infinite-board pieces
        PieceType::Rose => 450,
        PieceType::Huygen => 355,
    }
}


// ==================== Evaluation Constants ====================

// Infinite chess specific - enemy territory lines
const WHITE_ENEMY_LINE: i64 = 7;  // White pieces behind y > 7 are attacking
const BLACK_ENEMY_LINE: i64 = 2;  // Black pieces behind y < 2 are attacking

// King safety - ideal pawn shelter positions (like f2/f7 in standard chess)
// For white king at (5,1) or similar, pawns at (4,2), (5,2), (6,2) provide shelter
// For black king at (5,8) or similar, pawns at (4,7), (5,7), (6,7) provide shelter

// Bonuses/Penalties
const ROOK_BEHIND_ENEMY_BONUS: i32 = 30;      // Rook in enemy territory
const QUEEN_BEHIND_ENEMY_BONUS: i32 = 25;     // Queen in enemy territory
const PAWN_SHIELD_BONUS: i32 = 15;            // Pawn adjacent to king
const KNIGHT_CENTRALITY_BONUS: i32 = 10;      // Knight near center
const BISHOP_PAIR_BONUS: i32 = 30;            // Having both bishops
const ROOK_OPEN_FILE_BONUS: i32 = 25;         // Rook on file with no own pawns
const ROOK_SEMI_OPEN_BONUS: i32 = 15;         // Rook on file with only enemy pawns
const PASSED_PAWN_BONUS: i32 = 20;            // Passed pawn base bonus
const DOUBLED_PAWN_PENALTY: i32 = 15;         // Penalty for doubled pawns
const ISOLATED_PAWN_PENALTY: i32 = 10;        // Penalty for isolated pawns
#[allow(dead_code)]
const DEVELOPMENT_BONUS: i32 = 5;             // Piece moved from starting rank (future use)
const KING_TROPISM_BONUS: i32 = 3;            // Bonus per square closer to enemy king

// ==================== Main Evaluation ====================

pub fn evaluate(game: &GameState) -> i32 {
    // Check for insufficient material draw
    if is_insufficient_material(&game.board) {
        return 0;
    }
    
    // Start with material score
    let mut score = game.material_score;
    
    // Find king positions
    let (white_king, black_king) = find_kings(&game.board);
    
    // Check for endgame with lone king
    let white_only_king = is_lone_king(&game.board, PlayerColor::White);
    let black_only_king = is_lone_king(&game.board, PlayerColor::Black);
    
    // Handle lone king endgames - also works when one side has no king (practice positions)
    if black_only_king && black_king.is_some() {
        // White is winning (or has winning material) - add endgame bonus to help mate
        // Use white_king if available, otherwise use a dummy position for tropism
        let our_king = white_king.as_ref().cloned().unwrap_or_else(|| {
            // No white king - use center as reference for piece coordination
            Coordinate { x: 4, y: 4 }
        });
        score += evaluate_lone_king_endgame(game, &our_king, black_king.as_ref().unwrap(), PlayerColor::White);
    } else if white_only_king && white_king.is_some() {
        // Black is winning - add endgame bonus (negative for black advantage)
        let our_king = black_king.as_ref().cloned().unwrap_or_else(|| {
            Coordinate { x: 4, y: 4 }
        });
        score -= evaluate_lone_king_endgame(game, &our_king, white_king.as_ref().unwrap(), PlayerColor::Black);
    } else {
        // Normal game - use standard positional evaluation
        score += evaluate_pieces(game, &white_king, &black_king);
        score += evaluate_king_safety(game, &white_king, &black_king);
        score += evaluate_pawn_structure(game);
    }
    
    // Return from current player's perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

/// Fast evaluation for use in quiescence - just material + basic positional
#[allow(dead_code)]
pub fn evaluate_fast(game: &GameState) -> i32 {
    let score = game.material_score;
    
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

// ==================== Piece Evaluation ====================

fn evaluate_pieces(game: &GameState, white_king: &Option<Coordinate>, black_king: &Option<Coordinate>) -> i32 {
    let mut score: i32 = 0;
    
    let mut white_bishops = 0;
    let mut black_bishops = 0;
    
    for ((x, y), piece) in &game.board.pieces {
        let piece_score = match piece.piece_type {
            PieceType::Rook => evaluate_rook(game, *x, *y, piece.color, white_king, black_king),
            PieceType::Queen => evaluate_queen(*x, *y, piece.color, white_king, black_king),
            PieceType::Knight => evaluate_knight(*x, *y, piece.color, black_king, white_king),
            PieceType::Bishop => {
                if piece.color == PlayerColor::White {
                    white_bishops += 1;
                } else {
                    black_bishops += 1;
                }
                evaluate_bishop(*x, *y, piece.color)
            },
            PieceType::Pawn => evaluate_pawn_position(*x, *y, piece.color),
            _ => 0,
        };
        
        if piece.color == PlayerColor::White {
            score += piece_score;
        } else {
            score -= piece_score;
        }
    }
    
    // Bishop pair bonus
    if white_bishops >= 2 {
        score += BISHOP_PAIR_BONUS;
    }
    if black_bishops >= 2 {
        score -= BISHOP_PAIR_BONUS;
    }
    
    score
}

fn evaluate_rook(game: &GameState, x: i64, y: i64, color: PlayerColor, 
                 white_king: &Option<Coordinate>, black_king: &Option<Coordinate>) -> i32 {
    let mut bonus: i32 = 0;
    
    // Rook behind enemy lines bonus
    if color == PlayerColor::White && y > WHITE_ENEMY_LINE {
        bonus += ROOK_BEHIND_ENEMY_BONUS;
    } else if color == PlayerColor::Black && y < BLACK_ENEMY_LINE {
        bonus += ROOK_BEHIND_ENEMY_BONUS;
    }
    
    // Open/semi-open file bonus
    let (own_pawns_on_file, enemy_pawns_on_file) = count_pawns_on_file(game, x, color);
    if own_pawns_on_file == 0 {
        if enemy_pawns_on_file == 0 {
            bonus += ROOK_OPEN_FILE_BONUS;
        } else {
            bonus += ROOK_SEMI_OPEN_BONUS;
        }
    }
    
    // King tropism - closer to enemy king is better
    let enemy_king = if color == PlayerColor::White { black_king } else { white_king };
    if let Some(ek) = enemy_king {
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        bonus += ((20 - dist.min(20)) as i32) * KING_TROPISM_BONUS / 2;
    }
    
    bonus
}

fn evaluate_queen(x: i64, y: i64, color: PlayerColor,
                  white_king: &Option<Coordinate>, black_king: &Option<Coordinate>) -> i32 {
    let mut bonus: i32 = 0;
    
    // Queen behind enemy lines bonus
    if color == PlayerColor::White && y > WHITE_ENEMY_LINE {
        bonus += QUEEN_BEHIND_ENEMY_BONUS;
    } else if color == PlayerColor::Black && y < BLACK_ENEMY_LINE {
        bonus += QUEEN_BEHIND_ENEMY_BONUS;
    }
    
    // King tropism
    let enemy_king = if color == PlayerColor::White { black_king } else { white_king };
    if let Some(ek) = enemy_king {
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        bonus += ((15 - dist.min(15)) as i32) * KING_TROPISM_BONUS;
    }
    
    bonus
}

fn evaluate_knight(x: i64, y: i64, color: PlayerColor,
                   black_king: &Option<Coordinate>, white_king: &Option<Coordinate>) -> i32 {
    let mut bonus: i32 = 0;
    
    // Centrality bonus - knights are better in the center
    // For infinite chess, "center" is roughly around (4,4) to (4,5)
    let center_x: i64 = 4;
    let center_y: i64 = 4;
    let dist_to_center = (x - center_x).abs() + (y - center_y).abs();
    
    if dist_to_center <= 2 {
        bonus += KNIGHT_CENTRALITY_BONUS * 2;
    } else if dist_to_center <= 4 {
        bonus += KNIGHT_CENTRALITY_BONUS;
    }
    
    // King tropism - knights attacking near enemy king
    let enemy_king = if color == PlayerColor::White { black_king } else { white_king };
    if let Some(ek) = enemy_king {
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        if dist <= 3 {
            bonus += KING_TROPISM_BONUS * 3; // Knight fork potential
        }
    }
    
    bonus
}

fn evaluate_bishop(x: i64, y: i64, _color: PlayerColor) -> i32 {
    // Bishops are slightly better when not on the edge
    let mut bonus: i32 = 0;
    
    // Long diagonal control bonus
    if (x - y).abs() <= 1 || (x + y - 8).abs() <= 1 {
        bonus += 5; // On or near main diagonals
    }
    
    bonus
}

fn evaluate_pawn_position(x: i64, y: i64, color: PlayerColor) -> i32 {
    let mut bonus: i32 = 0;
    
    // Advancement bonus - more advanced pawns are better
    if color == PlayerColor::White {
        bonus += ((y - 2) as i32).max(0) * 3; // Bonus for ranks 3+
    } else {
        bonus += ((7 - y) as i32).max(0) * 3; // Bonus for ranks 6-
    }
    
    // Central pawns are valuable
    if x >= 3 && x <= 5 {
        bonus += 5;
    }
    
    bonus
}

// ==================== King Safety ====================

fn evaluate_king_safety(game: &GameState, white_king: &Option<Coordinate>, black_king: &Option<Coordinate>) -> i32 {
    let mut score: i32 = 0;
    
    // White king safety
    if let Some(wk) = white_king {
        score += evaluate_king_shelter(game, wk, PlayerColor::White);
    }
    
    // Black king safety
    if let Some(bk) = black_king {
        score -= evaluate_king_shelter(game, bk, PlayerColor::Black);
    }
    
    score
}

fn evaluate_king_shelter(game: &GameState, king: &Coordinate, color: PlayerColor) -> i32 {
    let mut safety: i32 = 0;
    
    // Count friendly pawns directly adjacent to king (8 squares around)
    for dx in -1..=1_i64 {
        for dy in -1..=1_i64 {
            if dx == 0 && dy == 0 { continue; }
            
            let check_x = king.x + dx;
            let check_y = king.y + dy;
            
            if let Some(piece) = game.board.get_piece(&check_x, &check_y) {
                if piece.piece_type == PieceType::Pawn && piece.color == color {
                    // Pawns in front are more valuable
                    if (color == PlayerColor::White && dy > 0) || 
                       (color == PlayerColor::Black && dy < 0) {
                        safety += PAWN_SHIELD_BONUS;
                    } else {
                        safety += PAWN_SHIELD_BONUS / 2; // Side/behind pawns less valuable
                    }
                }
            }
        }
    }
    
    // Penalty if king is too exposed (no pawns nearby)
    if safety == 0 {
        safety -= 15;
    }
    
    // Bonus for safe castled squares
    // White king on (6,2) or Black king on (6,7) are safe positions
    if color == PlayerColor::White {
        if king.x == 6 && king.y == 2 {
            safety += 20; // Ideal castled position
        } else if (king.x >= 6 && king.x <= 7) && king.y <= 2 {
            safety += 10; // Good castled area
        }
    } else {
        if king.x == 6 && king.y == 7 {
            safety += 20; // Ideal castled position
        } else if (king.x >= 6 && king.x <= 7) && king.y >= 7 {
            safety += 10; // Good castled area
        }
    }
    
    safety
}

// ==================== Pawn Structure ====================

fn evaluate_pawn_structure(game: &GameState) -> i32 {
    let mut score: i32 = 0;
    
    // Track pawns per file for each color
    let mut white_pawn_files: Vec<i64> = Vec::new();
    let mut black_pawn_files: Vec<i64> = Vec::new();
    let mut white_pawns: Vec<(i64, i64)> = Vec::new();
    let mut black_pawns: Vec<(i64, i64)> = Vec::new();
    
    for ((x, y), piece) in &game.board.pieces {
        if piece.piece_type == PieceType::Pawn {
            if piece.color == PlayerColor::White {
                white_pawn_files.push(*x);
                white_pawns.push((*x, *y));
            } else {
                black_pawn_files.push(*x);
                black_pawns.push((*x, *y));
            }
        }
    }
    
    // Doubled pawns penalty
    white_pawn_files.sort();
    black_pawn_files.sort();
    
    let mut prev_file: Option<i64> = None;
    for &file in &white_pawn_files {
        if prev_file == Some(file) {
            score -= DOUBLED_PAWN_PENALTY;
        }
        prev_file = Some(file);
    }
    
    prev_file = None;
    for &file in &black_pawn_files {
        if prev_file == Some(file) {
            score += DOUBLED_PAWN_PENALTY;
        }
        prev_file = Some(file);
    }
    
    // Passed pawn bonus
    for (x, y) in &white_pawns {
        if is_passed_pawn(*x, *y, PlayerColor::White, &black_pawns) {
            // More bonus for more advanced passed pawns
            score += PASSED_PAWN_BONUS + ((*y - 2) as i32).max(0) * 5;
        }
    }
    
    for (x, y) in &black_pawns {
        if is_passed_pawn(*x, *y, PlayerColor::Black, &white_pawns) {
            score -= PASSED_PAWN_BONUS + ((7 - *y) as i32).max(0) * 5;
        }
    }
    
    // Isolated pawn penalty
    for (x, _) in &white_pawns {
        let has_neighbor = white_pawns.iter().any(|(px, _)| (*px - *x).abs() == 1);
        if !has_neighbor {
            score -= ISOLATED_PAWN_PENALTY;
        }
    }
    
    for (x, _) in &black_pawns {
        let has_neighbor = black_pawns.iter().any(|(px, _)| (*px - *x).abs() == 1);
        if !has_neighbor {
            score += ISOLATED_PAWN_PENALTY;
        }
    }
    
    score
}

fn is_passed_pawn(x: i64, y: i64, color: PlayerColor, enemy_pawns: &[(i64, i64)]) -> bool {
    for (ex, ey) in enemy_pawns {
        // Check if enemy pawn is on same or adjacent file and ahead
        if (*ex - x).abs() <= 1 {
            if color == PlayerColor::White && *ey > y {
                return false;
            } else if color == PlayerColor::Black && *ey < y {
                return false;
            }
        }
    }
    true
}

// ==================== Helper Functions ====================

fn find_kings(board: &Board) -> (Option<Coordinate>, Option<Coordinate>) {
    let mut white_king: Option<Coordinate> = None;
    let mut black_king: Option<Coordinate> = None;
    
    for ((x, y), piece) in &board.pieces {
        if piece.piece_type == PieceType::King {
            if piece.color == PlayerColor::White {
                white_king = Some(Coordinate { x: *x, y: *y });
            } else {
                black_king = Some(Coordinate { x: *x, y: *y });
            }
        }
    }
    
    (white_king, black_king)
}

/// Check if a side only has a king (no other pieces)
fn is_lone_king(board: &Board, color: PlayerColor) -> bool {
    for (_, piece) in &board.pieces {
        if piece.color == color && piece.piece_type != PieceType::King {
            return false;
        }
    }
    true
}

/// Endgame evaluation when opponent only has a king
/// Rewards: pieces close to enemy king, kings close together, constraining enemy king
fn evaluate_lone_king_endgame(game: &GameState, our_king: &Coordinate, enemy_king: &Coordinate, winning_color: PlayerColor) -> i32 {
    let mut bonus: i32 = 0;
    
    // 1. Reward our pieces being close to enemy king (strong tropism)
    for ((x, y), piece) in &game.board.pieces {
        if piece.color == winning_color && piece.piece_type != PieceType::King {
            let dist = (x - enemy_king.x).abs() + (y - enemy_king.y).abs();
            // Closer is better - bonus decreases with distance
            let proximity_bonus = (20 - dist.min(20)) as i32 * 8;
            bonus += proximity_bonus;
        }
    }
    
    // 2. Reward our king being close to enemy king (helps in mating nets)
    let king_dist = (our_king.x - enemy_king.x).abs() + (our_king.y - enemy_king.y).abs();
    bonus += (14 - king_dist.min(14)) as i32 * 4;
    
    // 3. Reward constraining enemy king (pieces on same or adjacent rank/file)
    for ((x, y), piece) in &game.board.pieces {
        if piece.color == winning_color {
            let dx = (*x - enemy_king.x).abs();
            let dy = (*y - enemy_king.y).abs();

            // Same file or adjacent file on enemy king's rank (build "walls" above/below king)
            if dx == 0 && dy <= 1 {
                bonus += 10;
            }
            // Same rank or adjacent rank on enemy king's file (build "walls" left/right of king)
            if dy == 0 && dx <= 1 {
                bonus += 10;
            }
            // On or very near diagonals with enemy king (for bishops/queens)
            if dx == dy && dx <= 2 {
                bonus += 5;
            }
        }
    }
    
    bonus
}

fn count_pawns_on_file(game: &GameState, file: i64, color: PlayerColor) -> (i32, i32) {
    let mut own_pawns = 0;
    let mut enemy_pawns = 0;
    
    for ((x, _), piece) in &game.board.pieces {
        if *x == file && piece.piece_type == PieceType::Pawn {
            if piece.color == color {
                own_pawns += 1;
            } else {
                enemy_pawns += 1;
            }
        }
    }
    
    (own_pawns, enemy_pawns)
}

/// Check if a side has sufficient material to force checkmate in infinite chess.
/// In infinite chess without board edges, many piece combos that work in normal chess don't force mate.
/// Returns true if the side CAN potentially force checkmate.
pub fn has_sufficient_mating_material(board: &Board, color: PlayerColor, has_enemy_king: bool) -> bool {
    let mut queens = 0;
    let mut rooks = 0;
    let mut bishops = 0;
    let mut knights = 0;
    let mut chancellors = 0; // Rook + Knight
    let mut archbishops = 0; // Bishop + Knight
    let mut hawks = 0;
    let mut guards = 0;
    let mut pawns = 0;
    let mut light_bishops = 0;
    let mut dark_bishops = 0;
    
    for ((x, y), piece) in &board.pieces {
        if piece.color != color { continue; }
        match piece.piece_type {
            PieceType::Queen => queens += 1,
            PieceType::Rook => rooks += 1,
            PieceType::Bishop => {
                bishops += 1;
                // Check bishop color (light/dark square)
                if (x + y) % 2 == 0 {
                    light_bishops += 1;
                } else {
                    dark_bishops += 1;
                }
            },
            PieceType::Knight => knights += 1,
            PieceType::Chancellor => chancellors += 1,
            PieceType::Archbishop => archbishops += 1,
            PieceType::Hawk => hawks += 1,
            PieceType::Guard => guards += 1,
            PieceType::Pawn => pawns += 1,
            PieceType::Amazon => return true, // Amazon can always mate
            _ => {}
        }
    }
    
    // Based on infinite chess mating patterns (need to force mate WITH king help):
    
    // 2+ queens can force mate
    if queens >= 2 { return true; }
    
    // 2+ Rooks can force mate
    if rooks >= 2 { return true; }
    
    // 2+ Chancellors can force mate
    if chancellors >= 2 { return true; }
    
    // 2+ Archbishops can force mate
    if archbishops >= 2 { return true; }
    
    // Archbishop + Rook can mate
    if archbishops >= 1 && rooks >= 1 { return true; }
    
    // Chancellor + Bishop can mate
    if chancellors >= 1 && bishops >= 1 { return true; }
    
    // Chancellor + Archbishop can mate
    if chancellors >= 1 && archbishops >= 1 { return true; }
    
    // Rook + 2 minor pieces (bishop/knight)
    if rooks >= 1 && (bishops + knights) >= 2 { return true; }
    
    // Rook + 2+ Guards can mate
    if rooks >= 1 && guards >= 2 { return true; }
    
    // 4+ Bishops with both colors can mate
    if bishops >= 4 && light_bishops >= 2 && dark_bishops >= 2 { return true; }
    
    // 3 Bishops + Knight can mate
    if bishops >= 3 && knights >= 1 { return true; }
    
    // 2 Bishops + 2 Knights can mate
    if bishops >= 2 && knights >= 2 { return true; }
    
    // 2 Bishops + Guard can mate
    if bishops >= 2 && guards >= 1 { return true; }
    
    // 4+ Knights can mate
    if knights >= 4 { return true; }
    
    // 3+ Hawks can mate
    if hawks >= 3 { return true; }
    
    // 3+ Guards can mate
    if guards >= 3 { return true; }
    
    // Any major piece + pawn(s) that could promote
    if pawns >= 1 && (rooks >= 1 || chancellors >= 1 || archbishops >= 1) { return true; }
    
    // Multiple pawns that could promote to sufficient material
    if pawns >= 2 { return true; }
    
    false
}

/// Check if the game is a draw due to insufficient material
pub fn is_insufficient_material(board: &Board) -> bool {
    let white_has_king = board.pieces.iter().any(|(_, p)| p.piece_type == PieceType::King && p.color == PlayerColor::White);
    let black_has_king = board.pieces.iter().any(|(_, p)| p.piece_type == PieceType::King && p.color == PlayerColor::Black);
    
    let white_can_mate = has_sufficient_mating_material(board, PlayerColor::White, black_has_king);
    let black_can_mate = has_sufficient_mating_material(board, PlayerColor::Black, white_has_king);
    
    // Draw if neither side can mate
    !white_can_mate && !black_can_mate
}

pub fn calculate_initial_material(board: &Board) -> i32 {
    let mut score = 0;
    for (_, piece) in &board.pieces {
        let value = get_piece_value(piece.piece_type);
        if piece.color == PlayerColor::White {
            score += value;
        } else {
            score -= value;
        }
    }
    score
}
