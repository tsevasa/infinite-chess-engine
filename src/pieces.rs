use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerColor {
    White,
    Black,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Pawn,
    Knight,
    Hawk,
    King,
    Guard,
    Rook,
    Bishop,
    Queen,
    RoyalQueen,
    Chancellor,
    Archbishop,
    Amazon,
    Camel,
    Giraffe,
    Zebra,
    Knightrider,
    Centaur,
    RoyalCentaur,
    Huygen,
    Rose,
}

impl PieceType {
    pub fn get_value(&self) -> i32 {
        match self {
            PieceType::King => 0, // Cannot be captured, usually
            PieceType::Queen => 900,
            PieceType::Rook => 500,
            PieceType::Bishop => 300,
            PieceType::Knight => 300,
            PieceType::Pawn => 100,
            PieceType::Amazon => 1200, // Queen + Knight
            PieceType::Hawk => 500, // Leaper (2,0), (3,0), (2,2), (3,3) - approx Rook value?
            PieceType::Chancellor => 800, // Rook + Knight
            PieceType::Archbishop => 700, // Bishop + Knight
            PieceType::Centaur => 0, // King + Knight (Royal usually)
            PieceType::RoyalCentaur => 0, // Royal
            PieceType::Rose => 900, // Circular Nightrider - powerful
            PieceType::Camel => 200, // (3,1) leaper - Colorbound
            PieceType::Giraffe => 250, // (4,1) leaper
            PieceType::Zebra => 200, // (3,2) leaper
            PieceType::Knightrider => 900, // Multiple knight moves
            PieceType::Guard => 0, // King moves (non-royal) - approx 300? But usually royal in some variants. Let's say 300 if non-royal.
            PieceType::Huygen => 500, // Prime leaper - approx Rook?
            PieceType::RoyalQueen => 0, // Royal
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: PlayerColor,
    pub has_moved: bool,
}

impl Piece {
    pub fn new(piece_type: PieceType, color: PlayerColor) -> Self {
        Piece {
            piece_type,
            color,
            has_moved: false,
        }
    }
}
