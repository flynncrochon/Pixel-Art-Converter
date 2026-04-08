// Electron main process.
// Responsibilities:
//   1. Pick a free localhost port and spawn the Python FastAPI sidecar.
//   2. Wait for /health to come up.
//   3. Open the renderer window and tell it which port to use.
//   4. Kill the sidecar on quit.

const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');

Menu.setApplicationMenu(null);
const { spawn } = require('child_process');
const path = require('path');
const net = require('net');

let pyProc = null;
let pyPort = null;
let pyReady = false;

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function resolvePython() {
  // Allow override for venv users.
  if (process.env.PPA_PYTHON) return process.env.PPA_PYTHON;
  // Prefer a local .venv if present (created by `npm run setup`).
  const venvWin = path.join(__dirname, 'python', '.venv', 'Scripts', 'python.exe');
  const venvNix = path.join(__dirname, 'python', '.venv', 'bin', 'python');
  const fs = require('fs');
  if (fs.existsSync(venvWin)) return venvWin;
  if (fs.existsSync(venvNix)) return venvNix;
  return process.platform === 'win32' ? 'python' : 'python3';
}

async function waitForHealth(port, timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/health`);
      if (r.ok) return;
    } catch (_) { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`Python sidecar did not respond on port ${port} within ${timeoutMs}ms`);
}

async function startPython() {
  pyPort = await getFreePort();
  const py = resolvePython();
  const script = path.join(__dirname, 'python', 'server.py');
  console.log(`[main] spawning ${py} ${script} --port ${pyPort}`);

  pyProc = spawn(py, [script, '--port', String(pyPort)], {
    cwd: path.join(__dirname, 'python'),
    stdio: ['ignore', 'inherit', 'inherit'],
    windowsHide: true,
  });

  pyProc.on('exit', (code, sig) => {
    console.log(`[main] python sidecar exited code=${code} sig=${sig}`);
    pyReady = false;
  });

  await waitForHealth(pyPort);
  pyReady = true;
  console.log(`[main] sidecar ready on http://127.0.0.1:${pyPort}`);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 880,
    backgroundColor: '#1a1a1a',
    icon: path.join(__dirname, 'build', process.platform === 'win32' ? 'icon.ico' : 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  // win.webContents.openDevTools({ mode: 'detach' });
}

app.whenReady().then(async () => {
  try {
    await startPython();
  } catch (err) {
    dialog.showErrorBox(
      'Python sidecar failed to start',
      `${err.message}\n\nMake sure you ran:\n  cd python\n  python -m venv .venv\n  .venv\\Scripts\\activate\n  pip install -r requirements.txt`
    );
    app.quit();
    return;
  }

  ipcMain.handle('ppa:get-port', () => pyPort);
  ipcMain.handle('ppa:is-ready', () => pyReady);

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (pyProc && !pyProc.killed) {
    try { pyProc.kill(); } catch (_) {}
  }
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (pyProc && !pyProc.killed) {
    try { pyProc.kill(); } catch (_) {}
  }
});
