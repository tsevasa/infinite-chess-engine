use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub mod board;
pub mod game;
pub mod moves;
mod utils;
pub mod evaluation;
pub mod search;

use board::{Board, Piece, PieceType, PlayerColor, Coordinate};
use game::{GameState, EnPassantState};
use evaluation::calculate_initial_material;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    utils::set_panic_hook();
}

#[derive(Serialize, Deserialize)]
pub struct JsMove {
    pub from: String, // "x,y"
    pub to: String,   // "x,y"
    pub promotion: Option<String>,
}

#[derive(Deserialize)]
struct JsFullGame {
    board: JsBoard,
    turn: String,
    castling_rights: Vec<String>,
    en_passant: Option<JsEnPassant>,
    halfmove_clock: u32,
    fullmove_number: u32,
    #[serde(default)]
    move_history: Vec<JsMoveHistory>,
}

#[derive(Deserialize)]
struct JsMoveHistory {
    from: String,       // "x,y"
    to: String,         // "x,y"
    #[serde(default)]
    promotion: Option<String>,
}

#[derive(Deserialize)]
struct JsBoard {
    pieces: Vec<JsPiece>,
}

#[derive(Deserialize)]
struct JsPiece {
    x: String,
    y: String,
    piece_type: String,
    player: String,
}

#[derive(Deserialize)]
struct JsEnPassant {
    square: String,      // "x,y"
    pawn_square: String, // "x,y"
}

#[wasm_bindgen]
pub struct Engine {
    game: GameState,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(json_state: JsValue) -> Result<Engine, JsValue> {
        let js_game: JsFullGame = serde_wasm_bindgen::from_value(json_state)?;

        // Build GameState with i64 coordinates directly (no normalization)
        let mut board = Board::new();
        for p in &js_game.board.pieces {
            let x: i64 = p.x.parse().map_err(|_| JsValue::from_str("Invalid X coordinate"))?;
            let y: i64 = p.y.parse().map_err(|_| JsValue::from_str("Invalid Y coordinate"))?;

            let piece_type = match p.piece_type.as_str() {
                "p" => PieceType::Pawn,
                "n" => PieceType::Knight,
                "b" => PieceType::Bishop,
                "r" => PieceType::Rook,
                "q" => PieceType::Queen,
                "k" => PieceType::King,
                "g" => PieceType::Guard,
                "h" => PieceType::Hawk,
                "c" => PieceType::Chancellor,
                "a" => PieceType::Archbishop,
                "m" => PieceType::Amazon,
                "l" => PieceType::Camel,
                "i" => PieceType::Giraffe,
                "z" => PieceType::Zebra,
                "s" => PieceType::Knightrider,
                "e" => PieceType::Centaur,
                "d" => PieceType::RoyalCentaur,
                "o" => PieceType::Rose,
                "u" => PieceType::Huygen,
                "y" => PieceType::RoyalQueen,
                _ => PieceType::Pawn,
            };

            let color = match p.player.as_str() {
                "w" => PlayerColor::White,
                "b" => PlayerColor::Black,
                _ => PlayerColor::White,
            };

            board.set_piece(x, y, Piece::new(piece_type, color));
        }

        let turn = match js_game.turn.as_str() {
            "w" => PlayerColor::White,
            "b" => PlayerColor::Black,
            _ => PlayerColor::White,
        };

        // Parse castling rights directly as i64
        let mut castling_rights = HashSet::new();
        for cr in js_game.castling_rights {
            let parts: Vec<&str> = cr.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(x), Ok(y)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
                    castling_rights.insert(Coordinate::new(x, y));
                }
            }
        }

        // Parse en passant directly as i64
        let en_passant = if let Some(ep) = js_game.en_passant {
            let sq_parts: Vec<&str> = ep.square.split(',').collect();
            let pawn_parts: Vec<&str> = ep.pawn_square.split(',').collect();
            
            if sq_parts.len() == 2 && pawn_parts.len() == 2 {
                if let (Ok(sq_x), Ok(sq_y), Ok(pawn_x), Ok(pawn_y)) = (
                    sq_parts[0].parse::<i64>(),
                    sq_parts[1].parse::<i64>(),
                    pawn_parts[0].parse::<i64>(),
                    pawn_parts[1].parse::<i64>(),
                ) {
                    Some(EnPassantState {
                        square: Coordinate::new(sq_x, sq_y),
                        pawn_square: Coordinate::new(pawn_x, pawn_y),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let actual_fullmove = if js_game.fullmove_number <= 1 && !js_game.move_history.is_empty() {
            1 + (js_game.move_history.len() as u32 / 2)
        } else {
            js_game.fullmove_number
        };
        
        let mut game = GameState {
            board,
            turn,
            castling_rights,
            en_passant,
            halfmove_clock: js_game.halfmove_clock,
            fullmove_number: actual_fullmove,
            material_score: 0,
            hash_stack: Vec::with_capacity(128),
            null_moves: 0,
        };
        
        game.material_score = calculate_initial_material(&game.board);
        
        // Pre-populate hash_stack for move history
        for i in 0..js_game.move_history.len() {
            let dummy_hash = 0xFFFF_FFFF_0000_0000u64 ^ ((i as u64) * 0x9E3779B97F4A7C15);
            game.hash_stack.push(dummy_hash);
        }

        Ok(Engine { game })
    }

    pub fn get_best_move(&mut self) -> JsValue {
        if let Some(best_move) = search::get_best_move(&mut self.game, 50) {
            let js_move = JsMove {
                from: format!("{},{}", best_move.from.x, best_move.from.y),
                to: format!("{},{}", best_move.to.x, best_move.to.y),
                promotion: best_move.promotion.clone(),
            };
            serde_wasm_bindgen::to_value(&js_move).unwrap()
        } else {
            JsValue::NULL
        }
    }

    pub fn get_best_move_with_time(&mut self, time_limit_ms: u32) -> JsValue {
        if let Some(best_move) = search::get_best_move_timed(
            &mut self.game,
            50,
            time_limit_ms as u128,
            true,
        ) {
            let js_move = JsMove {
                from: format!("{},{}", best_move.from.x, best_move.from.y),
                to: format!("{},{}", best_move.to.x, best_move.to.y),
                promotion: best_move.promotion.clone(),
            };
            serde_wasm_bindgen::to_value(&js_move).unwrap()
        } else {
            JsValue::NULL
        }
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        self.game.perft(depth)
    }
    
    pub fn setup_standard_chess(&mut self) {
        self.game.setup_standard_chess();
    }

    /// Returns all legal moves as a JS array of {from: "x,y", to: "x,y", promotion: string|null}
    pub fn get_legal_moves_js(&mut self) -> JsValue {
        let pseudo_legal = self.game.get_legal_moves();
        let mut legal_moves: Vec<JsMove> = Vec::new();

        for m in pseudo_legal {
            let undo = self.game.make_move(&m);
            let illegal = self.game.is_move_illegal();
            self.game.undo_move(&m, undo);

            if !illegal {
                legal_moves.push(JsMove {
                    from: format!("{},{}", m.from.x, m.from.y),
                    to: format!("{},{}", m.to.x, m.to.y),
                    promotion: m.promotion.clone(),
                });
            }
        }

        serde_wasm_bindgen::to_value(&legal_moves).unwrap_or(JsValue::NULL)
    }
}
