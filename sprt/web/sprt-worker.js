import initOld, { Engine as EngineOld } from './pkg-old/hydrochess_wasm.js';
import initNew, { Engine as EngineNew } from './pkg-new/hydrochess_wasm.js';

let wasmReady = false;

function getStandardPosition() {
    const pieces = [];
    pieces.push({ x: '1', y: '1', piece_type: 'r', player: 'w' });
    pieces.push({ x: '2', y: '1', piece_type: 'n', player: 'w' });
    pieces.push({ x: '3', y: '1', piece_type: 'b', player: 'w' });
    pieces.push({ x: '4', y: '1', piece_type: 'q', player: 'w' });
    pieces.push({ x: '5', y: '1', piece_type: 'k', player: 'w' });
    pieces.push({ x: '6', y: '1', piece_type: 'b', player: 'w' });
    pieces.push({ x: '7', y: '1', piece_type: 'n', player: 'w' });
    pieces.push({ x: '8', y: '1', piece_type: 'r', player: 'w' });
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '2', piece_type: 'p', player: 'w' });
    }
    pieces.push({ x: '1', y: '8', piece_type: 'r', player: 'b' });
    pieces.push({ x: '2', y: '8', piece_type: 'n', player: 'b' });
    pieces.push({ x: '3', y: '8', piece_type: 'b', player: 'b' });
    pieces.push({ x: '4', y: '8', piece_type: 'q', player: 'b' });
    pieces.push({ x: '5', y: '8', piece_type: 'k', player: 'b' });
    pieces.push({ x: '6', y: '8', piece_type: 'b', player: 'b' });
    pieces.push({ x: '7', y: '8', piece_type: 'n', player: 'b' });
    pieces.push({ x: '8', y: '8', piece_type: 'r', player: 'b' });
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '7', piece_type: 'p', player: 'b' });
    }

    // Standard infinite-chess special rights: all pawns (double-step)
    // plus kings and rooks (castling and related king/rook rights).
    const special_rights = [];
    for (let i = 1; i <= 8; i++) {
        special_rights.push(i + ',2'); // white pawns
        special_rights.push(i + ',7'); // black pawns
    }
    // White rooks and king
    special_rights.push('1,1');
    special_rights.push('8,1');
    special_rights.push('5,1');
    // Black rooks and king
    special_rights.push('1,8');
    special_rights.push('8,8');
    special_rights.push('5,8');

    return {
        board: { pieces },
        // Starting side for the game; the WASM engine will reconstruct the
        // current side-to-move by replaying move_history.
        turn: 'w',
        // Support both old and new APIs: legacy castling_rights for old EngineOld
        // builds, and special_rights for the new engine.
        castling_rights: [],
        special_rights,
        en_passant: null,
        halfmove_clock: 0,
        fullmove_number: 1,
        move_history: [],
        game_rules: null,
        world_bounds: null,
    };
}

function applyMove(position, move) {
    const pieces = position.board.pieces;
    const [fromX, fromY] = move.from.split(',');
    const [toX, toY] = move.to.split(',');

    const capturedIdx = pieces.findIndex(p => p.x === toX && p.y === toY);
    if (capturedIdx !== -1) {
        pieces.splice(capturedIdx, 1);
    }

    const movingPiece = pieces.find(p => p.x === fromX && p.y === fromY);
    if (!movingPiece) {
        throw new Error('No piece at ' + move.from);
    }

    // Enforce side-to-move: do not allow engines to move the opponent's pieces.
    if (position.turn === 'w' && movingPiece.player !== 'w') {
        throw new Error('Illegal move: white to move but piece at ' + move.from + ' is not white');
    }
    if (position.turn === 'b' && movingPiece.player !== 'b') {
        throw new Error('Illegal move: black to move but piece at ' + move.from + ' is not black');
    }

    movingPiece.x = toX;
    movingPiece.y = toY;

    if (move.promotion) {
        movingPiece.piece_type = move.promotion.toLowerCase();
    }

    position.turn = position.turn === 'w' ? 'b' : 'w';
    return position;
}

function isGameOver(position) {
    const kings = position.board.pieces.filter(p => p.piece_type === 'k');
    if (kings.length < 2) {
        return { over: true, reason: 'checkmate' };
    }
    if (position.board.pieces.length <= 2) {
        return { over: true, reason: 'draw' };
    }
    return { over: false };
}

function clonePosition(position) {
    // Simple deep clone for our small position objects
    return JSON.parse(JSON.stringify(position));
}

// Simple material-only evaluation (centipawns from White's perspective),
// used for smart adjudication when the position is clearly decided.
function evaluateMaterial(position) {
    const pieces = position.board.pieces;
    let score = 0;
    for (const p of pieces) {
        const t = p.piece_type.toLowerCase();
        let v = 0;
        if (t === 'p') v = 100;
        else if (t === 'n') v = 250;
        else if (t === 'b') v = 450;
        else if (t === 'r') v = 650;
        else if (t === 'q') v = 1350;
        if (p.player === 'w') score += v; else score -= v;
    }
    return score;
}

async function ensureInit() {
    if (!wasmReady) {
        await initOld();
        await initNew();
        wasmReady = true;
    }
}

async function playSingleGame(timePerMove, maxMoves, newPlaysWhite, openingMove, materialThreshold) {
    const startPosition = getStandardPosition();
    let position = clonePosition(startPosition);
    const newColor = newPlaysWhite ? 'w' : 'b';
    const moveLines = [];
    const moveHistory = [];

    // Apply opening move if provided (always white's first move)
    if (openingMove) {
        moveLines.push('W: ' + openingMove.from + '>' + openingMove.to);
        position = applyMove(position, openingMove);
        moveHistory.push({
            from: openingMove.from,
            to: openingMove.to,
            promotion: openingMove.promotion || null
        });
    }

    for (let i = 0; i < maxMoves; i++) {
        const sideToMove = position.turn;
        const isWhiteTurn = sideToMove === 'w';

        // Build full-game object for engines: immutable start position + move history.
        const gameInput = clonePosition(startPosition);
        gameInput.move_history = moveHistory.slice();

        // Smart adjudication: if the position is completely decided by simple
        // material count, stop early and award the game.
        const evalCp = evaluateMaterial(position);
        const threshold = typeof materialThreshold === 'number' ? materialThreshold : 0;
        if (threshold > 0 && Math.abs(evalCp) >= threshold) {
            const winningColor = evalCp > 0 ? 'w' : 'b';
            const result = winningColor === newColor ? 'win' : 'loss';
            const winnerStr = winningColor === 'w' ? 'White' : 'Black';
            moveLines.push('# Game adjudicated by material: ~' + (evalCp > 0 ? '+' : '') + evalCp + ' cp for ' + winnerStr + ' (threshold ' + threshold + ' cp)');
            moveLines.push('# Engines: new=' + (newColor === 'w' ? 'White' : 'Black') + ', old=' + (newColor === 'w' ? 'Black' : 'White'));
            return { result, log: moveLines.join('\n'), reason: 'material_adjudication', materialThreshold: threshold };
        }

        // Otherwise, let the appropriate engine choose a move on this gameInput
        const EngineClass = isWhiteTurn
            ? (newPlaysWhite ? EngineNew : EngineOld)
            : (newPlaysWhite ? EngineOld : EngineNew);
        const engineName = isWhiteTurn
            ? (newPlaysWhite ? 'new' : 'old')
            : (newPlaysWhite ? 'old' : 'new');

        const engine = new EngineClass(gameInput);
        const move = engine.get_best_move_with_time(timePerMove);
        engine.free();

        if (!move || !move.from || !move.to) {
            // Engine failed to produce a move: treat as that engine losing.
            moveLines.push('# Engine ' + (engineName === 'new' ? 'HydroChess New' : 'HydroChess Old') +
                ' failed to return a move.');
            const result = engineName === 'new' ? 'loss' : 'win';
            return { result, log: moveLines.join('\n') };
        }

        // First try to apply the move to our local position. If this fails,
        // we treat it as an illegal move from the engine: the side to move
        // loses immediately, and we DO NOT record this move in the log or
        // move_history so that the resulting ICN is always playable.
        try {
            position = applyMove(position, move);
        } catch (e) {
            // Illegal move from the engine: side that moved loses. Do NOT
            // record the move itself in history so ICN remains playable.
            moveLines.push('# Illegal move from ' + (engineName === 'new' ? 'HydroChess New' : 'HydroChess Old') +
                ': ' + (move && move.from && move.to ? (move.from + '>' + move.to) : 'null') +
                ' (' + (e && e.message ? e.message : String(e)) + ')');
            const result = engineName === 'new' ? 'loss' : 'win';
            return { result, log: moveLines.join('\n') };
        }

        // Only after a successful apply do we log and record the move.
        moveLines.push(
            (sideToMove === 'w' ? 'W' : 'B') + ': ' + move.from + '>' + move.to +
            (move.promotion ? '=' + move.promotion : '')
        );

        // Track move history from the initial position for subsequent engine calls
        moveHistory.push({
            from: move.from,
            to: move.to,
            promotion: move.promotion || null
        });

        const gameState = isGameOver(position);
        if (gameState.over) {
            if (gameState.reason === 'draw') {
                return { result: 'draw', log: moveLines.join('\n') };
            }
            const result = sideToMove === newColor ? 'win' : 'loss';
            return { result, log: moveLines.join('\n') };
        }
    }

    return { result: 'draw', log: moveLines.join('\n') };
}

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'runGame') {
        try {
            await ensureInit();
            const { result, log, reason, materialThreshold } = await playSingleGame(
                msg.timePerMove,
                msg.maxMoves,
                msg.newPlaysWhite,
                msg.openingMove,
                msg.materialThreshold,
            );
            self.postMessage({
                type: 'result',
                gameIndex: msg.gameIndex,
                result,
                log,
                newPlaysWhite: msg.newPlaysWhite,
                reason: reason || null,
                materialThreshold: materialThreshold ?? msg.materialThreshold ?? null,
            });
        } catch (err) {
            self.postMessage({
                type: 'error',
                gameIndex: msg.gameIndex,
                error: err.message || String(err),
            });
        }
    }
};
