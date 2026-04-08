// Bridge between renderer and main process. Exposes a tiny, safe API.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('ppa', {
  getPort: () => ipcRenderer.invoke('ppa:get-port'),
  isReady: () => ipcRenderer.invoke('ppa:is-ready'),
});
