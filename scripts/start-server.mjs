import { spawn, execSync } from 'child_process';
import { writeFileSync, readFileSync, existsSync, openSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const pidFile = resolve(root, '.server.pid');
const logFile = resolve(root, 'server.log');

// Resolve the venv python so we don't depend on shell activation
const venvPy = resolve(
  root,
  '.venv',
  process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
);
const py = existsSync(venvPy) ? venvPy : (process.platform === 'win32' ? 'python' : 'python3');

// ── 1. Build frontend ────────────────────────────────────────────────
console.log('Building frontend...');
execSync('npm run build', { cwd: resolve(root, 'frontend'), stdio: 'inherit' });

// ── 2. Stop any already-running server ───────────────────────────────
if (existsSync(pidFile)) {
  try {
    const oldPid = parseInt(readFileSync(pidFile, 'utf8').trim());
    if (process.platform === 'win32') {
      execSync(`taskkill /F /PID ${oldPid} /T`, { stdio: 'ignore' });
    } else {
      process.kill(oldPid, 'SIGTERM');
    }
  } catch { /* already gone */ }
}

// ── 3. Launch waitress in the background ─────────────────────────────
const out = openSync(logFile, 'w');
const err = openSync(logFile, 'a');

const child = spawn(py, ['serve.py'], {
  cwd: root,
  detached: true,
  stdio: ['ignore', out, err],
  env: { ...process.env },
});
child.unref();

writeFileSync(pidFile, String(child.pid));

console.log('');
console.log('  PhotoFinder started -> http://localhost:5000');
console.log(`  PID:  ${child.pid}`);
console.log('  Log:  server.log');
console.log('  Stop: npm run stop:local');
console.log('');
