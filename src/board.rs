use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Coordinate {
    pub x: i64,
    pub y: i64,
}

impl Coordinate {
    pub fn new(x: i64, y: i64) -> Self {
        Coordinate { x, y }
    }

    pub fn get_x(&self) -> i64 {
        self.x
    }

    pub fn get_y(&self) -> i64 {
        self.y
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayerColor {
    White,
    Black,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    Guard,
    Camel,
    Giraffe,
    Zebra,
    Knightrider,
    Amazon,
    RoyalQueen,
    Hawk,
    Chancellor,
    Archbishop,
    Centaur,
    RoyalCentaur,
    Rose,
    Huygen,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: PlayerColor,
}

impl Piece {
    pub fn new(piece_type: PieceType, color: PlayerColor) -> Self {
        Piece { piece_type, color }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Board {
    pub pieces: HashMap<(i64, i64), Piece>,
}

impl Board {
    pub fn new() -> Self {
        Board {
            pieces: HashMap::new(),
        }
    }

    pub fn set_piece(&mut self, x: i64, y: i64, piece: Piece) {
        self.pieces.insert((x, y), piece);
    }

    pub fn get_piece(&self, x: &i64, y: &i64) -> Option<&Piece> {
        self.pieces.get(&(*x, *y))
    }

    pub fn remove_piece(&mut self, x: &i64, y: &i64) -> Option<Piece> {
        self.pieces.remove(&(*x, *y))
    }
}
