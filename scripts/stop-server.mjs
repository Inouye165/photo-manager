import { readFileSync, unlinkSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const pidFile = resolve(root, '.server.pid');

if (!existsSync(pidFile)) {
  console.log('Server not running (no .server.pid file).');
  process.exit(0);
}

const pid = readFileSync(pidFile, 'utf8').trim();

try {
  if (process.platform === 'win32') {
    execSync(`taskkill /F /PID ${pid} /T`, { stdio: 'ignore' });
  } else {
    process.kill(parseInt(pid), 'SIGTERM');
  }
  console.log(`Server stopped (PID ${pid}).`);
} catch {
  console.log(`Could not kill PID ${pid} — server may have already stopped.`);
}

try { unlinkSync(pidFile); } catch { /* already removed */ }
