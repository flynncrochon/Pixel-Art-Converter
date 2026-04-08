// Bridge between renderer and main process. Exposes a tiny, safe API.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('ppa', {
  getPort: () => ipcRenderer.invoke('ppa:get-port'),
  isReady: () => ipcRenderer.invoke('ppa:is-ready'),
  pickInputFolder: () => ipcRenderer.invoke('ppa:pick-input-folder'),
  pickOutputFolder: () => ipcRenderer.invoke('ppa:pick-output-folder'),
  readImageB64: (filePath) => ipcRenderer.invoke('ppa:read-image-b64', filePath),
  saveImageB64: (filePath, b64) => ipcRenderer.invoke('ppa:save-image-b64', filePath, b64),
});
