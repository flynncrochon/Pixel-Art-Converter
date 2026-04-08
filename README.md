# Pixel Art Converter

Pixel Art Converter is a small desktop app that turns regular images into pixel art with a live preview. It's an Electron front-end wrapped around the [proper-pixel-art](https://pypi.org/project/proper-pixel-art/) Python library, with a FastAPI sidecar handling the heavy lifting.

## Installation

**Warning** You'll need Python and Node installed before the app is useful:

- **Python 3.10+**. The app spawns a local FastAPI sidecar on startup.
- **Node 18+**. Needed for Electron.

### Run from source

```
npm install
npm run setup:py        # or `npm run setup:py:posix` on macOS/Linux
npm start
```

`setup:py` creates a virtualenv under `python/.venv` and installs the requirements (`proper-pixel-art`, `fastapi`, `uvicorn`, `pillow`, `pydantic`). If you'd rather use your own interpreter, set `PPA_PYTHON` to its path before launching.

### Desktop app

Grab the latest build from GitHub Releases, or package it yourself:

```
npm run pack
```

## Guide

### Source panel

Drop an image onto the sidebar or hit **Choose file…** to load it. The original sits in the **Source** pane on the left and the converted pixel art renders in the **Pixel art** pane on the right. Anything you tweak in the controls re-renders the right pane on the fly.

### Background and masking

- **Transparent background** keeps alpha through the pipeline instead of flattening to a solid color.
- **Mask color → transparent** picks a key color (with an adjustable **Tolerance** slider) and knocks it out of the source. Handy for sprite rips on flat backdrops.
- **Remove edge speckles** cleans up the lone bright pixels that creep in around alpha edges.

### Pixel size

**Manual pixel width** sets how many source pixels collapse into a single output cell. Leave it at `0` (auto) and the library detects the grid for you; bump it up when you want a specific block size. `1` gives you a true 1:1 pass-through.

### Palette

This is where you decide what the output is allowed to look like.

- **Size** controls how many colors the auto-palette is allowed to use.
- **Preset** picks a fixed retro palette — Game Boy, NES, PICO-8, CGA, Sweetie 16, Endesga 32, and friends. Selecting a preset overrides the size slider.
- **Saturation** scales the chroma of the final palette before it's applied.
- **Dithering (Floyd–Steinberg)** trades flat fills for stippled gradients.
- **Import palette from image…** lets you yank colors out of any reference image and use them as a custom palette. **Clear** drops back to auto.

The chosen palette swatches show up underneath so you can sanity-check what's in play.

### Outline

- **Black outline** wraps the sprite in a 1-pixel border.
- **Include diagonals** controls whether the outline tracks corners or only orthogonal neighbors.
- **Thickness** bumps the border up to 8 pixels.

### Saving and selection

When you're happy, click **Save output…** to write the pixel art to disk. Use **Select region** on the right pane to crop a sub-rectangle of the result before saving — useful when you only want a single sprite out of a sheet.
