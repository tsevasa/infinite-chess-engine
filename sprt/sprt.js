#!/usr/bin/env node
/**
 * HydroChess Web SPRT Helper
 *
 * - Treats root/pkg as the OLD engine snapshot
 * - Builds a NEW web WASM into root/pkg-new
 * - Copies both into sprt/web/pkg-old and sprt/web/pkg-new
 * - Starts `npx serve .` in sprt/web so browser UI can import both
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

const SPRT_DIR = __dirname;
const PROJECT_ROOT = path.join(SPRT_DIR, '..');

// Root-level packages
const ROOT_PKG_OLD = path.join(PROJECT_ROOT, 'pkg');      // existing old snapshot
const ROOT_PKG_NEW = path.join(PROJECT_ROOT, 'pkg-new');  // new web build

// Web UI directories
const WEB_DIR = path.join(SPRT_DIR, 'web');
const WEB_PKG_OLD_DIR = path.join(WEB_DIR, 'pkg-old');
const WEB_PKG_NEW_DIR = path.join(WEB_DIR, 'pkg-new');

function copyDirectory(src, dest) {
    if (!fs.existsSync(src)) {
        throw new Error(`Source directory does not exist: ${src}`);
    }
    if (!fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
    }
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);
        if (entry.isDirectory()) {
            copyDirectory(srcPath, destPath);
        } else {
            fs.copyFileSync(srcPath, destPath);
        }
    }
}

function rmDirIfExists(dir) {
    if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

function snapshotOldFromRoot() {
    if (!fs.existsSync(ROOT_PKG_OLD)) {
        console.error('[web-sprt] Error: expected OLD pkg directory at', ROOT_PKG_OLD);
        console.error('[web-sprt] Build your old reference version with: wasm-pack build --target web --out-dir pkg');
        process.exit(1);
    }

    if (!fs.existsSync(WEB_DIR)) {
        fs.mkdirSync(WEB_DIR, { recursive: true });
    }

    console.log('[web-sprt] Copying OLD pkg (root/pkg) to sprt/web/pkg-old');
    rmDirIfExists(WEB_PKG_OLD_DIR);
    copyDirectory(ROOT_PKG_OLD, WEB_PKG_OLD_DIR);
}

function buildNewWebPkg() {
    console.log('\n[web-sprt] Building NEW WASM (target=web -> pkg-new)...');
    try {
        execSync('wasm-pack build --target web --out-dir pkg-new', {
            cwd: PROJECT_ROOT,
            stdio: 'inherit',
        });
    } catch (e) {
        console.error('[web-sprt] wasm-pack build failed:', e.message);
        process.exit(1);
    }
    if (!fs.existsSync(ROOT_PKG_NEW)) {
        console.error('[web-sprt] Build finished but pkg-new is missing:', ROOT_PKG_NEW);
        process.exit(1);
    }
}

function syncNewToWeb() {
    if (!fs.existsSync(WEB_DIR)) {
        fs.mkdirSync(WEB_DIR, { recursive: true });
    }

    console.log('[web-sprt] Copying NEW pkg (root/pkg-new) to sprt/web/pkg-new');
    rmDirIfExists(WEB_PKG_NEW_DIR);
    copyDirectory(ROOT_PKG_NEW, WEB_PKG_NEW_DIR);
}

function startServer() {
    console.log('[web-sprt] Starting dev server in sprt/web (npx serve .)');
    console.log('[web-sprt] Open this URL in your browser (default): http://localhost:3000');

    const child = spawn('npx', ['serve', '.'], {
        cwd: WEB_DIR,
        stdio: 'inherit',
        shell: true, // avoid EINVAL on Windows by using shell resolution for npx
    });

    child.on('exit', (code) => {
        process.exit(code ?? 0);
    });
}

(function main() {
    try {
        snapshotOldFromRoot();
        buildNewWebPkg();
        syncNewToWeb();
        startServer();
    } catch (e) {
        console.error('[web-sprt] Fatal error:', e.message);
        process.exit(1);
    }
})();
