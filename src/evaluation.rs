use crate::board::{PieceType, PlayerColor, Board, Coordinate};
use crate::game::GameState;

// ==================== Piece Values ====================

pub fn get_piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        // neutral/blocking pieces - no material value
        PieceType::Void => 0,
        PieceType::Obstacle => 0,
        
        // orthodox - adjusted for infinite chess where sliders dominate
        PieceType::Pawn => 100,
        PieceType::Knight => 250,   // Weak in infinite chess - limited range
        PieceType::Bishop => 450,   // Strong slider - worth knight + 1.5 pawns
        PieceType::Rook => 650,     // Very strong in infinite chess
        PieceType::Queen | PieceType::RoyalQueen => 1350, // > 2 rooks
        PieceType::King | PieceType::Guard => 220,

        // short / medium range
        PieceType::Camel => 270,   // (1,3) leaper
        PieceType::Giraffe => 260, // (1,4) leaper
        PieceType::Zebra => 260,   // (2,3) leaper

        // riders / compounds
        PieceType::Knightrider => 700,
        PieceType::Amazon => 1550,
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

// Bonuses/Penalties for infinite chess
const BEHIND_KING_BONUS: i32 = 40;            // Major piece past enemy king's rank/file
const PAWN_SHIELD_BONUS: i32 = 20;            // Pawn adjacent to king
const KNIGHT_NEAR_KING_BONUS: i32 = 15;       // Knight protecting own king
const BISHOP_PAIR_BONUS: i32 = 50;            // Having both bishops (strong in infinite chess)
const ROOK_OPEN_FILE_BONUS: i32 = 25;         // Rook on file with no own pawns
const ROOK_SEMI_OPEN_BONUS: i32 = 15;         // Rook on file with only enemy pawns
const PASSED_PAWN_BONUS: i32 = 8;             // Passed pawn base bonus (reduced for infinite chess)
const DOUBLED_PAWN_PENALTY: i32 = 3;          // Penalty for doubled pawns (minimal in infinite chess)
const ISOLATED_PAWN_PENALTY: i32 = 2;         // Penalty for isolated pawns (minimal in infinite chess)
const KING_TROPISM_BONUS: i32 = 4;            // Bonus per square closer to enemy king

// Starting squares for pieces (standard chess layout)
// White back rank is y=1, Black back rank is y=8
const WHITE_BACK_RANK: i64 = 1;
const BLACK_BACK_RANK: i64 = 8;

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
    let mut white_bishop_colors: (bool, bool) = (false, false); // (light, dark)
    let mut black_bishop_colors: (bool, bool) = (false, false);
    
    for ((x, y), piece) in &game.board.pieces {
        let mut piece_score = match piece.piece_type {
            PieceType::Rook => evaluate_rook(game, *x, *y, piece.color, white_king, black_king),
            PieceType::Queen => evaluate_queen(*x, *y, piece.color, white_king, black_king),
            PieceType::Knight => evaluate_knight(*x, *y, piece.color, black_king, white_king),
            PieceType::Bishop => {
                if piece.color == PlayerColor::White {
                    white_bishops += 1;
                    if (*x + *y) % 2 == 0 { white_bishop_colors.0 = true; } 
                    else { white_bishop_colors.1 = true; }
                } else {
                    black_bishops += 1;
                    if (*x + *y) % 2 == 0 { black_bishop_colors.0 = true; } 
                    else { black_bishop_colors.1 = true; }
                }
                evaluate_bishop(*x, *y, piece.color, white_king, black_king)
            },
            PieceType::Pawn => evaluate_pawn_position(*x, *y, piece.color),
            _ => 0,
        };
        
        // Development penalty: major pieces on starting rank
        let back_rank = if piece.color == PlayerColor::White { WHITE_BACK_RANK } else { BLACK_BACK_RANK };
        if *y == back_rank {
            let penalty = match piece.piece_type {
                PieceType::Queen => 25,  // Queen should develop early
                PieceType::Rook => 15,   // Rooks should activate
                PieceType::Bishop => 12, // Bishops should develop
                PieceType::Knight => 5,  // Knights less important
                _ => 0,
            };
            piece_score -= penalty;
        }
        
        if piece.color == PlayerColor::White {
            score += piece_score;
        } else {
            score -= piece_score;
        }
    }
    
    // Bishop pair bonus - even stronger if opposite colors
    if white_bishops >= 2 {
        score += BISHOP_PAIR_BONUS;
        if white_bishop_colors.0 && white_bishop_colors.1 {
            score += 20; // Extra bonus for opposite colored bishops
        }
    }
    if black_bishops >= 2 {
        score -= BISHOP_PAIR_BONUS;
        if black_bishop_colors.0 && black_bishop_colors.1 {
            score -= 20;
        }
    }
    
    score
}

fn evaluate_rook(game: &GameState, x: i64, y: i64, color: PlayerColor, 
                 white_king: &Option<Coordinate>, black_king: &Option<Coordinate>) -> i32 {
    let mut bonus: i32 = 0;
    
    // Behind enemy king bonus - rook past the enemy king's rank/file for attack
    let enemy_king = if color == PlayerColor::White { black_king } else { white_king };
    if let Some(ek) = enemy_king {
        // Behind enemy king on Y axis (rank)
        if color == PlayerColor::White && y > ek.y {
            bonus += BEHIND_KING_BONUS;
        } else if color == PlayerColor::Black && y < ek.y {
            bonus += BEHIND_KING_BONUS;
        }
        // On same file as enemy king or adjacent - attack potential
        if (x - ek.x).abs() <= 1 {
            bonus += 15;
        }
        // King tropism
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        bonus += ((20 - dist.min(20)) as i32) * KING_TROPISM_BONUS / 2;
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
    
    bonus
}

fn evaluate_queen(x: i64, y: i64, color: PlayerColor,
                  white_king: &Option<Coordinate>, black_king: &Option<Coordinate>) -> i32 {
    let mut bonus: i32 = 0;
    
    // Behind enemy king bonus - queen past the enemy king for attack/mate threats
    let enemy_king = if color == PlayerColor::White { black_king } else { white_king };
    if let Some(ek) = enemy_king {
        // Behind enemy king on Y axis
        if color == PlayerColor::White && y > ek.y {
            bonus += BEHIND_KING_BONUS + 10; // Queen gets extra bonus
        } else if color == PlayerColor::Black && y < ek.y {
            bonus += BEHIND_KING_BONUS + 10;
        }
        // King tropism - queen very strong near enemy king
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        bonus += ((15 - dist.min(15)) as i32) * KING_TROPISM_BONUS;
    }
    
    bonus
}

fn evaluate_knight(x: i64, y: i64, color: PlayerColor,
                   black_king: &Option<Coordinate>, white_king: &Option<Coordinate>) -> i32 {
    let mut bonus: i32 = 0;
    
    // Knights are weak in infinite chess - minimal positional bonus
    // Small bonus for being near friendly king (defensive)
    let own_king = if color == PlayerColor::White { white_king } else { black_king };
    if let Some(ok) = own_king {
        let dist = (x - ok.x).abs() + (y - ok.y).abs();
        if dist <= 3 {
            bonus += KNIGHT_NEAR_KING_BONUS; // Knight protecting king
        } else if dist <= 5 {
            bonus += KNIGHT_NEAR_KING_BONUS / 2;
        }
    }
    
    // Small bonus for being near enemy king (fork potential)
    let enemy_king = if color == PlayerColor::White { black_king } else { white_king };
    if let Some(ek) = enemy_king {
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        if dist <= 3 {
            bonus += 10; // Fork potential
        }
    }
    
    bonus
}

fn evaluate_bishop(x: i64, y: i64, color: PlayerColor, 
                   white_king: &Option<Coordinate>, black_king: &Option<Coordinate>) -> i32 {
    let mut bonus: i32 = 0;
    
    // Long diagonal control bonus
    if (x - y).abs() <= 1 || (x + y - 8).abs() <= 1 {
        bonus += 8; // On or near main diagonals
    }
    
    // Behind enemy king bonus - bishop past enemy king for attack
    let enemy_king = if color == PlayerColor::White { black_king } else { white_king };
    if let Some(ek) = enemy_king {
        if color == PlayerColor::White && y > ek.y {
            bonus += BEHIND_KING_BONUS / 2; // Bishops get half bonus (less direct than rooks)
        } else if color == PlayerColor::Black && y < ek.y {
            bonus += BEHIND_KING_BONUS / 2;
        }
        // King tropism
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        bonus += ((15 - dist.min(15)) as i32) * KING_TROPISM_BONUS / 2;
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

/// Endgame evaluation when opponent only has a lone king
/// Key strategy for 2+ rooks: SANDWICH the king
/// - Rooks on adjacent ranks (e.g., rank 8 and 10 if king on rank 9)
/// - Rooks protect each other (same file)
/// - This cuts off both sides, king can't escape or attack rooks
/// - Then bring our king in to help deliver mate
fn evaluate_lone_king_endgame(game: &GameState, our_king: &Coordinate, enemy_king: &Coordinate, winning_color: PlayerColor) -> i32 {
    let mut bonus: i32 = 0;

    let king_needed = needs_king_for_mate(&game.board, winning_color);

    // Collect slider positions
    struct SliderInfo { x: i64, y: i64 }
    let mut sliders: Vec<SliderInfo> = Vec::new();

    for ((x, y), piece) in &game.board.pieces {
        if piece.color != winning_color || piece.piece_type.is_royal() { continue; }
        match piece.piece_type {
            PieceType::Rook | PieceType::Chancellor | PieceType::Queen | PieceType::Amazon => {
                sliders.push(SliderInfo { x: *x, y: *y });
            }
            _ => {}
        }
    }

    // ========== SANDWICH DETECTION ==========
    // The ideal formation: two rooks on ADJACENT ranks/files to the enemy king
    // Example: enemy king on rank 9, rooks on ranks 8 and 10 (same file to protect each other)
    
    let mut has_sandwich_horizontal = false; // Rooks on files adjacent to king's file (king sandwiched on files)
    let mut has_sandwich_vertical = false;   // Rooks on ranks adjacent to king's rank (king sandwiched on ranks)
    
    // Collect ranks/files above/below and left/right of the enemy king
    let ranks_above: Vec<i64> = sliders.iter().filter(|s| s.y > enemy_king.y).map(|s| s.y).collect();
    let ranks_below: Vec<i64> = sliders.iter().filter(|s| s.y < enemy_king.y).map(|s| s.y).collect();
    let files_right: Vec<i64> = sliders.iter().filter(|s| s.x > enemy_king.x).map(|s| s.x).collect();
    let files_left: Vec<i64> = sliders.iter().filter(|s| s.x < enemy_king.x).map(|s| s.x).collect();

    // Closest fences in each direction as Option<i64>, reusable for box/run geometry
    let closest_above = ranks_above.iter().min().copied();
    let closest_below = ranks_below.iter().max().copied();
    let closest_right = files_right.iter().min().copied();
    let closest_left = files_left.iter().max().copied();

    // Rank sandwich (rooks above and below king's rank)
    if let (Some(ca), Some(cb)) = (closest_above, closest_below) {
        has_sandwich_vertical = true;
        let gap = ca - cb - 1; // Gap the king is confined to
        
        // MASSIVE bonus for tight sandwich
        if gap <= 1 {
            bonus += 600; // King trapped to 1 rank
        } else if gap <= 2 {
            bonus += 450;
        } else if gap <= 3 {
            bonus += 350;
        } else if gap <= 5 {
            bonus += 250;
        } else {
            bonus += 150;
        }
    }
    
    // File sandwich (rooks left and right of king's file)
    if let (Some(cr), Some(cl)) = (closest_right, closest_left) {
        has_sandwich_horizontal = true;
        let gap = cr - cl - 1;
        
        if gap <= 1 {
            bonus += 600;
        } else if gap <= 2 {
            bonus += 450;
        } else if gap <= 3 {
            bonus += 350;
        } else if gap <= 5 {
            bonus += 250;
        } else {
            bonus += 150;
        }
    }

    // ========== ROOK MUTUAL PROTECTION ==========
    // Rooks on the same rank OR same file protect each other
    // This is CRITICAL - enemy king can't attack protected rooks
    
    let mut protected_count = 0;
    for i in 0..sliders.len() {
        let mut is_protected = false;
        for j in 0..sliders.len() {
            if i != j {
                if sliders[i].x == sliders[j].x || sliders[i].y == sliders[j].y {
                    is_protected = true;
                    break;
                }
            }
        }
        if is_protected {
            protected_count += 1;
        }
    }
    bonus += protected_count as i32 * 150; // Big bonus for each protected rook

    // ========== FENCE CLOSENESS ==========
    // Bonus for having rooks close to the enemy king (but not ON same rank/file)
    
    for s in &sliders {
        let rank_dist = (s.y - enemy_king.y).abs();
        let file_dist = (s.x - enemy_king.x).abs();
        
        // Rank fence quality (rook NOT on same rank as king)
        if s.y != enemy_king.y {
            if rank_dist == 1 {
                bonus += 200; // Perfect fence - 1 rank away
            } else if rank_dist == 2 {
                bonus += 150;
            } else if rank_dist <= 4 {
                bonus += 100;
            } else if rank_dist <= 6 {
                bonus += 50;
            } else {
                bonus += 20;
            }
        }
        
        // File fence quality
        if s.x != enemy_king.x {
            if file_dist == 1 {
                bonus += 200;
            } else if file_dist == 2 {
                bonus += 150;
            } else if file_dist <= 4 {
                bonus += 100;
            } else if file_dist <= 6 {
                bonus += 50;
            } else {
                bonus += 20;
            }
        }
    }

    // ========== STRONG PENALTY FOR CHECKING ==========
    // Rook on same rank/file as king = giving check, NOT fencing
    // This breaks the sandwich and lets king escape
    
    for s in &sliders {
        if s.x == enemy_king.x {
            let dist = (s.y - enemy_king.y).abs();
            if dist > 1 {
                bonus -= 250; // Very bad - checking instead of fencing
            } else if dist == 1 {
                bonus -= 50; // Adjacent might be part of mate, small penalty
            }
        }
        if s.y == enemy_king.y {
            let dist = (s.x - enemy_king.x).abs();
            if dist > 1 {
                bonus -= 250;
            } else if dist == 1 {
                bonus -= 50;
            }
        }
    }

    // ========== CUTTING OFF ESCAPE DIRECTION ==========
    // Bonus for having fences on the side AWAY from our king
    // This pushes enemy king toward our king
    
    let our_dx = our_king.x - enemy_king.x;
    let our_dy = our_king.y - enemy_king.y;
    
    // If our king is to the right, we want fences on the left to push enemy right
    if our_dx > 0 && !files_left.is_empty() {
        bonus += 200;
    }
    if our_dx < 0 && !files_right.is_empty() {
        bonus += 200;
    }
    if our_dy > 0 && !ranks_below.is_empty() {
        bonus += 200;
    }
    if our_dy < 0 && !ranks_above.is_empty() {
        bonus += 200;
    }

    // ========== KING INVOLVEMENT ==========
    let king_dist = (our_king.x - enemy_king.x).abs() + (our_king.y - enemy_king.y).abs();
    let dx_abs = our_dx.abs();
    let dy_abs = our_dy.abs();

    // General escape geometry using run distances in all four directions.
    // We measure how many squares the enemy king can run in each cardinal
    // direction before hitting a fence (or effectively "infinity" if there is
    // no fence). If the king can only run a very small number of squares in
    // the directions that increase distance from our king, we treat it as
    // boxed in and start bringing our king in aggressively.

    // Distances to the nearest fence in each direction (or a large value if none)
    let run_up = closest_above.map(|f| f - enemy_king.y - 1).unwrap_or(1000);
    let run_down = closest_below.map(|f| enemy_king.y - f - 1).unwrap_or(1000);
    let run_right = closest_right.map(|f| f - enemy_king.x - 1).unwrap_or(1000);
    let run_left = closest_left.map(|f| enemy_king.x - f - 1).unwrap_or(1000);

    // Small grace: allow the king to shuffle a couple of squares in the
    // "escape" directions without counting it as a real escape. This fixes
    // the jankiness where the king steps just past our king's file/rank but
    // is still effectively trapped in the same box.
    const RUN_GRACE: i64 = 2;

    // Horizontal run in the direction that increases distance from our king
    let run_away_h = if our_dx > 0 {
        // Our king is to the right, so running left increases distance
        run_left
    } else if our_dx < 0 {
        // Our king is to the left, so running right increases distance
        run_right
    } else {
        // Same file: any horizontal direction that has the longer run
        run_left.max(run_right)
    };

    // Vertical run in the direction that increases distance from our king
    let run_away_v = if our_dy > 0 {
        // Our king is above, so running down increases distance
        run_down
    } else if our_dy < 0 {
        // Our king is below, so running up increases distance
        run_up
    } else {
        // Same rank: take the longer vertical run
        run_up.max(run_down)
    };

    let enemy_can_run_away = run_away_h > RUN_GRACE || run_away_v > RUN_GRACE;

    // Also check if rooks are protected - if so, king can't harass them
    let rooks_protected = protected_count >= 2 || (protected_count >= 1 && sliders.len() >= 2);

    if king_needed {
        // Bring king in if:
        // 1. Enemy can't run away more than a couple of squares in the
        //    directions that increase distance from our king (boxed in), or
        // 2. Rooks are protected so king harassment doesn't matter, or
        // 3. We have a strong sandwich in either axis.
        let should_approach = !enemy_can_run_away || rooks_protected || has_sandwich_horizontal || has_sandwich_vertical;
        
        if should_approach {
            // Enemy is trapped - bring king in AGGRESSIVELY.
            // Make king approach dominate over any small rook shuffles once the
            // box/net is effectively closed.
            let prox = (30 - king_dist.min(30)) as i32;
            bonus += prox * 40; // Stronger approach bonus so K-moves beat rook shuffles
            
            // Opposition bonus
            if (dx_abs == 2 && dy_abs == 0) || (dx_abs == 0 && dy_abs == 2) {
                bonus += 250;
            }
            // Very close
            if dx_abs <= 2 && dy_abs <= 2 {
                bonus += 150;
            }
            if dx_abs <= 1 && dy_abs <= 1 && (dx_abs + dy_abs) > 0 {
                bonus += 100;
            }
        } else {
            // Enemy can still escape - mild king approach, focus on fencing
            let prox = (20 - king_dist.min(20)) as i32;
            bonus += prox * 5;
        }
    } else {
        // King not needed (3+ sliders) - small tiebreaker
        let prox = (15 - king_dist.min(15)) as i32;
        bonus += prox * 2;
    }

    // ========== SLIDER COUNT BONUS ==========
    if sliders.len() >= 2 {
        bonus += 100;
    }
    if sliders.len() >= 3 {
        bonus += 150;
    }

    // ========== SHORT-RANGE PIECE MOBILIZATION ==========
    // Reward short-range pieces (king, knights, etc.) for moving closer to the
    // enemy king, especially when we don't yet have enough sliders to build a
    // perfect box. This fixes cases like Q+2N vs K where the knights drift
    // away because only the slider heuristics are active.
    let few_sliders = sliders.len() <= 1;
    // Stronger base when there is only a single slider (e.g. Q+2N vs K), so
    // the evaluation really wants the short-range pieces to participate.
    let short_base_scale: i32 = if few_sliders { 6 } else { 3 };

    for ((px, py), piece) in &game.board.pieces {
        if piece.color != winning_color {
            continue;
        }

        // Skip long-range sliders here; they are already handled by the
        // sandwich/fence logic above.
        let is_slider_piece = matches!(
            piece.piece_type,
            PieceType::Rook
                | PieceType::Chancellor
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Bishop
                | PieceType::Amazon
                | PieceType::Knightrider
                | PieceType::Huygen
        );

        if is_slider_piece {
            continue;
        }

        // King proximity is already handled by the king-involvement logic
        // above; avoid double-counting it here.
        if piece.piece_type == PieceType::King {
            continue;
        }

        let dist = (px - enemy_king.x).abs() + (py - enemy_king.y).abs();
        let capped = dist.min(12);
        let prox = (12 - capped) as i32; // 1 step closer = positive bonus, far away ~ 0
        if prox <= 0 {
            continue;
        }

        // Scale more strongly for true short-range attackers.
        let piece_scale: i32 = match piece.piece_type {
            PieceType::Guard | PieceType::Centaur | PieceType::RoyalCentaur => 3,
            PieceType::Knight | PieceType::Camel | PieceType::Giraffe | PieceType::Zebra => 3,
            PieceType::Hawk | PieceType::Rose => 2,
            _ => 1,
        };

        bonus += prox * short_base_scale * piece_scale;
    }

    bonus
}

/// Determine if king is needed for mate based on material
/// Based on the spreadsheet: "Forced Mate with King?" column
fn needs_king_for_mate(board: &Board, color: PlayerColor) -> bool {
    let mut queens = 0;
    let mut rooks = 0;
    let mut bishops = 0;
    let mut knights = 0;
    let mut chancellors = 0;
    let mut archbishops = 0;
    let mut amazons = 0;
    let mut hawks = 0;
    let mut guards = 0;
    
    for (_, piece) in &board.pieces {
        if piece.color != color { continue; }
        match piece.piece_type {
            PieceType::Queen | PieceType::RoyalQueen => queens += 1,
            PieceType::Rook => rooks += 1,
            PieceType::Bishop => bishops += 1,
            PieceType::Knight => knights += 1,
            PieceType::Chancellor => chancellors += 1,
            PieceType::Archbishop => archbishops += 1,
            PieceType::Amazon => amazons += 1,
            PieceType::Hawk => hawks += 1,
            PieceType::Guard => guards += 1,
            _ => {}
        }
    }
    
    // Cases where king is NOT needed (can mate without king)
    // 3+ Rooks, 2+ Chancellors, 2+ Queens, Amazon, etc.
    if rooks >= 3 { return false; }
    if chancellors >= 2 { return false; }
    if queens >= 2 { return false; }
    if amazons >= 1 { return false; }
    if archbishops >= 3 { return false; }
    if hawks >= 4 { return false; }
    if bishops >= 6 { return false; }
    
    // Strong combinations that don't need king
    if queens >= 1 && chancellors >= 1 { return false; }
    if queens >= 1 && bishops >= 2 { return false; }
    if queens >= 1 && knights >= 2 { return false; }
    if queens >= 1 && guards >= 2 { return false; }
    if queens >= 1 && rooks >= 1 && (bishops >= 1 || knights >= 1) { return false; }
    if chancellors >= 1 && bishops >= 2 { return false; }
    if rooks >= 2 && (bishops >= 2 || knights >= 2 || guards >= 1) { return false; }
    if rooks >= 1 && bishops >= 3 { return false; }
    if rooks >= 1 && knights >= 4 { return false; }
    if rooks >= 1 && guards >= 2 { return false; }
    
    // Default: king is needed
    true
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
/// Based on the official insufficientmaterial.ts from infinitechess.org
/// 
/// Logic: This lists INSUFFICIENT scenarios. Anything not matching
/// those scenarios is sufficient. We check if the current material fits within
/// any insufficient scenario.
pub fn has_sufficient_mating_material(board: &Board, color: PlayerColor, has_our_king: bool) -> bool {
    let mut queens = 0;
    let mut rooks = 0;
    let mut bishops = 0;
    let mut knights = 0;
    let mut chancellors = 0;
    let mut archbishops = 0;
    let mut hawks = 0;
    let mut guards = 0;
    let mut pawns = 0;
    let mut amazons = 0;
    let mut knightriders = 0;
    let mut huygens = 0;
    let mut light_bishops = 0;
    let mut dark_bishops = 0;
    
    for ((x, y), piece) in &board.pieces {
        if piece.color != color { continue; }
        match piece.piece_type {
            PieceType::Queen | PieceType::RoyalQueen => queens += 1,
            PieceType::Rook => rooks += 1,
            PieceType::Bishop => {
                bishops += 1;
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
            PieceType::Amazon => amazons += 1,
            PieceType::Knightrider => knightriders += 1,
            PieceType::Huygen => huygens += 1,
            _ => {}
        }
    }
    
    // Amazon can always mate
    if amazons >= 1 { return true; }
    
    // Helper: check if we have "only" certain pieces (nothing else)
    let has_only = |q: i32, r: i32, b: i32, n: i32, c: i32, a: i32, h: i32, g: i32, p: i32, s: i32, hu: i32| -> bool {
        queens <= q && rooks <= r && bishops <= b && knights <= n &&
        chancellors <= c && archbishops <= a && hawks <= h && guards <= g &&
        pawns <= p && knightriders <= s && huygens <= hu
    };
    
    // =====================================================================
    // 1K vs 1k scenarios (with our king helping)
    // These are INSUFFICIENT scenarios from insuffmatScenarios_1K1k
    // =====================================================================
    if has_our_king {
        // {queensW: 1} - single queen insufficient
        if queens == 1 && has_only(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {bishopsW: [Inf, 1]} - any number of same-color bishops insufficient
        if bishops > 0 && (light_bishops == 0 || dark_bishops == 0) && 
           has_only(0, 0, i32::MAX, 0, 0, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {knightsW: 3} - up to 3 knights insufficient
        if knights <= 3 && has_only(0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {hawksW: 2} - 2 hawks insufficient
        if hawks <= 2 && has_only(0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0) { return false; }
        
        // {hawksW: 1, bishopsW: [1,0]} - hawk + same-color bishop
        if hawks == 1 && bishops == 1 && (light_bishops == 0 || dark_bishops == 0) &&
           has_only(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0) { return false; }
        
        // {rooksW: 1, knightsW: 1} - rook + knight insufficient
        if rooks == 1 && knights == 1 && has_only(0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {rooksW: 1, bishopsW: [1,0]} - rook + same-color bishop
        if rooks == 1 && bishops == 1 && (light_bishops == 0 || dark_bishops == 0) &&
           has_only(0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {rooksW: 1, rooksB: 1} - rook vs rook (but we only count our pieces, so skip)
        
        // {archbishopsW: 1, bishopsW: [1,0]} - archbishop + same-color bishop
        if archbishops == 1 && bishops == 1 && (light_bishops == 0 || dark_bishops == 0) &&
           has_only(0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0) { return false; }
        
        // {archbishopsW: 1, knightsW: 1} - archbishop + knight
        if archbishops == 1 && knights == 1 && has_only(0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0) { return false; }
        
        // {knightsW: 1, bishopsW: [Inf, 0]} - knight + any same-color bishops
        if knights == 1 && bishops > 0 && (light_bishops == 0 || dark_bishops == 0) &&
           has_only(0, 0, i32::MAX, 1, 0, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {knightsW: 1, bishopsW: [1,1]} - knight + one of each bishop color
        if knights == 1 && light_bishops == 1 && dark_bishops == 1 &&
           has_only(0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {knightsW: 2, bishopsW: [1,0]} - 2 knights + same-color bishop
        if knights == 2 && bishops == 1 && (light_bishops == 0 || dark_bishops == 0) &&
           has_only(0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {guardsW: 1} - single guard insufficient
        if guards == 1 && has_only(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0) { return false; }
        
        // {chancellorsW: 1} - single chancellor insufficient
        if chancellors == 1 && has_only(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0) { return false; }
        
        // {knightridersW: 2} - 2 knightriders insufficient
        if knightriders <= 2 && has_only(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0) { return false; }
        
        // {pawnsW: 3} - up to 3 pawns insufficient
        if pawns <= 3 && has_only(0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0) { return false; }
        
        // Everything else with king is sufficient
        return true;
    }
    
    // =====================================================================
    // 0K vs 1k scenarios (without our king)
    // These are INSUFFICIENT scenarios from insuffmatScenarios_0K1k
    // Anything NOT in this list is sufficient
    // =====================================================================
    
    // {queensW: 1, rooksW: 1}
    if queens == 1 && rooks == 1 && has_only(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {queensW: 1, knightsW: 1}
    if queens == 1 && knights == 1 && has_only(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {queensW: 1, bishopsW: [1,0]}
    if queens == 1 && bishops == 1 && (light_bishops == 0 || dark_bishops == 0) &&
       has_only(1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {queensW: 1, pawnsW: 1}
    if queens == 1 && pawns == 1 && has_only(1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0) { return false; }
    
    // {bishopsW: [2,2]} - 2 light + 2 dark bishops
    if light_bishops == 2 && dark_bishops == 2 && has_only(0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {bishopsW: [Inf, 1]} - any number of one color + 1 of other
    if bishops > 0 && (light_bishops == 0 || dark_bishops == 0 || 
       (light_bishops <= 1 && dark_bishops > 0) || (dark_bishops <= 1 && light_bishops > 0)) &&
       has_only(0, 0, i32::MAX, 0, 0, 0, 0, 0, 0, 0, 0) {
        // Actually: [Inf, 1] means unlimited of one color, at most 1 of the other
        if (light_bishops == 0 || dark_bishops <= 1) && (dark_bishops == 0 || light_bishops <= 1) {
            return false;
        }
    }
    
    // {knightsW: 4}
    if knights <= 4 && has_only(0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {knightsW: 2, bishopsW: [Inf, 0]}
    if knights <= 2 && bishops > 0 && (light_bishops == 0 || dark_bishops == 0) &&
       has_only(0, 0, i32::MAX, 2, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {knightsW: 2, bishopsW: [1,1]}
    if knights <= 2 && light_bishops == 1 && dark_bishops == 1 &&
       has_only(0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {knightsW: 1, bishopsW: [2,1]}
    if knights == 1 && light_bishops <= 2 && dark_bishops <= 1 && bishops <= 3 &&
       has_only(0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {hawksW: 3}
    if hawks <= 3 && has_only(0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0) { return false; }
    
    // {rooksW: 1, knightsW: 1, bishopsW: [1,0]}
    if rooks == 1 && knights == 1 && bishops == 1 && (light_bishops == 0 || dark_bishops == 0) &&
       has_only(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {rooksW: 1, knightsW: 1, pawnsW: 1}
    if rooks == 1 && knights == 1 && pawns == 1 && has_only(0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0) { return false; }
    
    // {rooksW: 1, knightsW: 2}
    if rooks == 1 && knights <= 2 && has_only(0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {rooksW: 1, guardsW: 1}
    if rooks == 1 && guards == 1 && has_only(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0) { return false; }
    
    // {rooksW: 2, bishopsW: [1,0]}
    if rooks == 2 && bishops == 1 && (light_bishops == 0 || dark_bishops == 0) &&
       has_only(0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {rooksW: 2, knightsW: 1}
    if rooks == 2 && knights == 1 && has_only(0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {rooksW: 2, pawnsW: 1}
    if rooks == 2 && pawns == 1 && has_only(0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0) { return false; }
    
    // {archbishopsW: 1, bishopsW: [2,0]}
    if archbishops == 1 && bishops <= 2 && (light_bishops == 0 || dark_bishops == 0) &&
       has_only(0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0) { return false; }
    
    // {archbishopsW: 1, bishopsW: [1,1]}
    if archbishops == 1 && light_bishops == 1 && dark_bishops == 1 &&
       has_only(0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0) { return false; }
    
    // {archbishopsW: 1, knightsW: 2}
    if archbishops == 1 && knights <= 2 && has_only(0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0) { return false; }
    
    // {archbishopsW: 2}
    if archbishops <= 2 && has_only(0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0) { return false; }
    
    // {chancellorsW: 1, guardsW: 1}
    if chancellors == 1 && guards == 1 && has_only(0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0) { return false; }
    
    // {chancellorsW: 1, knightsW: 1}
    if chancellors == 1 && knights == 1 && has_only(0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {chancellorsW: 1, rooksW: 1}
    if chancellors == 1 && rooks == 1 && has_only(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0) { return false; }
    
    // {guardsW: 2}
    if guards <= 2 && has_only(0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0) { return false; }
    
    // {amazonsW: 1} - amazon is always sufficient
    // (already handled above)
    
    // {knightridersW: 3}
    if knightriders <= 3 && has_only(0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0) { return false; }
    
    // {pawnsW: 6}
    if pawns <= 6 && has_only(0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0) { return false; }
    
    // {huygensW: 4}
    if huygens <= 4 && has_only(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4) { return false; }
    
    // Everything else is sufficient (not in any insufficient scenario)
    true
}

/// Check if the game is a draw due to insufficient material
pub fn is_insufficient_material(board: &Board) -> bool {
    // Count pieces quickly - if too many pieces, definitely not insufficient
    let total_pieces = board.pieces.len();
    if total_pieces >= 10 { return false; } // Fast exit for complex positions
    
    let white_has_king = board.pieces.iter().any(|(_, p)| p.piece_type.is_royal() && p.color == PlayerColor::White);
    let black_has_king = board.pieces.iter().any(|(_, p)| p.piece_type.is_royal() && p.color == PlayerColor::Black);
    
    let white_can_mate = has_sufficient_mating_material(board, PlayerColor::White, white_has_king);
    let black_can_mate = has_sufficient_mating_material(board, PlayerColor::Black, black_has_king);
    
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
