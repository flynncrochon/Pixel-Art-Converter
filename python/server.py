"""FastAPI sidecar that exposes proper-pixel-art's pixelate() over HTTP.

Spawned by the Electron main process. Listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from proper_pixel_art import colors as ppa_colors, utils as ppa_utils
from proper_pixel_art.pixelate import pixelate


def block_pixelate(image: Image.Image, pixel_width: int,
                   num_colors: int | None) -> Image.Image:
    """Pixel-perfect block downsample: pack N x N source pixels into one cell.

    Bypasses proper_pixel_art's mesh-detection pipeline (which silently
    upscales 2x and crops the border, so a 108x139 input at pixel_width=1
    came out 211x273). Instead, when the user explicitly sets a pixel width
    we just walk the source on an N-pixel grid:

      - pixel_width=1 -> output dims == input dims (true 1:1)
      - pixel_width=N -> ceil(W/N) x ceil(H/N)

    Cell colors are picked with the same helpers the library uses, so the
    look matches the auto-pixel-width path.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    width, height = image.size
    pw = max(1, int(pixel_width))
    rgba_arr = np.array(image)

    if num_colors is not None:
        quantized = ppa_colors.palette_img(image, num_colors=num_colors)
        rgb_arr = np.array(quantized.convert("RGB"))
    else:
        rgb_arr = None

    # 1:1 fast path — no shrinking, just optionally apply quantized RGB.
    if pw == 1:
        if rgb_arr is None:
            return image
        out = np.dstack([rgb_arr, rgba_arr[..., 3]])
        return Image.fromarray(out, mode="RGBA")

    out_w = (width + pw - 1) // pw
    out_h = (height + pw - 1) // pw
    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    for j in range(out_h):
        y0 = j * pw
        y1 = min(y0 + pw, height)
        for i in range(out_w):
            x0 = i * pw
            x1 = min(x0 + pw, width)
            if rgb_arr is None:
                cell = rgba_arr[y0:y1, x0:x1]
                out[j, i] = ppa_colors.get_cell_color_skip_quantization(cell)
            else:
                cell_rgb = rgb_arr[y0:y1, x0:x1]
                cell_a = rgba_arr[y0:y1, x0:x1, 3]
                out[j, i] = ppa_colors.get_cell_color_with_alpha(cell_rgb, cell_a)

    return Image.fromarray(out, mode="RGBA")


def remove_edge_speckles(img: Image.Image, threshold: int = 215,
                         max_blob_size: int = 6) -> Image.Image:
    """Remove halo speckles by deleting small connected blobs of bright pixels.

    Strategy:
      1. Build a mask of opaque, near-white pixels (RGB >= `threshold`).
      2. Run 8-connected component labelling on that mask.
      3. Wipe any blob whose area is <= `max_blob_size` to transparent.

    A real white feature (snowman body, paper, etc.) is a large connected
    region and survives. AI-upscaler halo dots are 1- to 4-pixel blobs and
    get removed regardless of whether they touch the subject silhouette.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img)

    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    is_opaque = a > 0
    is_white = (r >= threshold) & (g >= threshold) & (b >= threshold) & is_opaque

    if not is_white.any():
        return img

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        is_white.astype(np.uint8), connectivity=8,
    )

    # Label 0 is the background (everything not white). Find labels whose
    # blob area is small and mark every pixel with that label for removal.
    sizes = stats[:, cv2.CC_STAT_AREA]
    small_labels = np.where(sizes <= max_blob_size)[0]
    small_labels = small_labels[small_labels != 0]
    if small_labels.size == 0:
        return img

    remove_mask = np.isin(labels, small_labels)
    arr[remove_mask] = (0, 0, 0, 0)
    return Image.fromarray(arr, "RGBA")


def add_outline(img: Image.Image, thickness: int = 1,
                color: tuple[int, int, int] = (0, 0, 0),
                diagonal: bool = True) -> Image.Image:
    """Add a solid outline around the opaque region of an RGBA image.

    Dilates the alpha mask by `thickness` pixels and paints the newly-added
    border pixels with `color`. Pixels that were already opaque are kept.
    """
    if thickness <= 0:
        return img
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3]
    opaque = (alpha > 0).astype(np.uint8)

    k = 2 * thickness + 1
    if diagonal:
        # 8-connected: full square kernel includes diagonal neighbours.
        kernel = np.ones((k, k), np.uint8)
    else:
        # 4-connected: plus-shaped kernel skips diagonals so corners stay open.
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    dilated = cv2.dilate(opaque, kernel, iterations=1)

    border = (dilated > 0) & (opaque == 0)
    if not border.any():
        return img
    arr[border] = (color[0], color[1], color[2], 255)
    return Image.fromarray(arr, "RGBA")

# ---- Palette presets (RGB tuples) ----
PALETTE_PRESETS: dict[str, list[tuple[int, int, int]]] = {
    "gameboy": [
        (15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15),
    ],
    "gameboy_pocket": [
        (0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255),
    ],
    "pico8": [
        (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
        (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
        (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
        (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170),
    ],
    "cga": [
        (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
        (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
        (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
        (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
    ],
    "sweetie16": [
        (26, 28, 44), (93, 39, 93), (177, 62, 83), (239, 125, 87),
        (255, 205, 117), (167, 240, 112), (56, 183, 100), (37, 113, 121),
        (41, 54, 111), (59, 93, 201), (65, 166, 246), (115, 239, 247),
        (244, 244, 244), (148, 176, 194), (86, 108, 134), (51, 60, 87),
    ],
    "endesga32": [
        (190, 74, 47), (215, 118, 67), (228, 166, 114), (255, 206, 145),
        (242, 160, 87), (250, 105, 0), (234, 73, 9), (193, 33, 21),
        (115, 23, 45), (74, 27, 57), (39, 23, 49), (10, 10, 13),
        (47, 64, 76), (76, 99, 110), (107, 132, 138), (159, 185, 184),
        (220, 234, 222), (255, 255, 255), (244, 213, 60), (151, 200, 67),
        (51, 152, 75), (38, 109, 79), (37, 73, 75), (44, 165, 191),
        (61, 186, 218), (90, 220, 240), (203, 219, 252), (104, 134, 197),
        (75, 78, 168), (55, 41, 121), (95, 30, 109), (160, 49, 122),
    ],
    # NES is a longer palette — provide a representative subset.
    "nes": [
        (124, 124, 124), (0, 0, 252), (0, 0, 188), (68, 40, 188),
        (148, 0, 132), (168, 0, 32), (168, 16, 0), (136, 20, 0),
        (80, 48, 0), (0, 120, 0), (0, 104, 0), (0, 88, 0),
        (0, 64, 88), (0, 0, 0), (188, 188, 188), (0, 120, 248),
        (0, 88, 248), (104, 68, 252), (216, 0, 204), (228, 0, 88),
        (248, 56, 0), (228, 92, 16), (172, 124, 0), (0, 184, 0),
        (0, 168, 0), (0, 168, 68), (0, 136, 136), (248, 248, 248),
        (60, 188, 252), (104, 136, 252), (152, 120, 248), (248, 120, 248),
        (248, 88, 152), (248, 120, 88), (252, 160, 68), (248, 184, 0),
        (184, 248, 24), (88, 216, 84), (88, 248, 152), (0, 232, 216),
        (120, 120, 120), (252, 252, 252), (164, 228, 252), (184, 184, 248),
        (216, 184, 248), (248, 184, 248), (248, 164, 192), (240, 208, 176),
        (252, 224, 168), (248, 216, 120), (216, 248, 120), (184, 248, 184),
        (184, 248, 216), (0, 252, 252), (248, 216, 248),
    ],
}


def _palette_to_pil(palette: list[tuple[int, int, int]]) -> Image.Image:
    """Build a PIL P-mode palette image from a list of RGB tuples."""
    n = len(palette)
    if n < 1:
        raise ValueError("palette must have at least 1 color")
    flat: list[int] = []
    for r, g, b in palette:
        flat.extend((int(r), int(g), int(b)))
    # PIL palettes need 256 entries; pad by repeating the first color.
    pad = palette[0]
    while len(flat) < 256 * 3:
        flat.extend((int(pad[0]), int(pad[1]), int(pad[2])))
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat[: 256 * 3])
    return pal_img


def adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    """Scale RGB saturation by `factor` (1.0 = no change). Alpha untouched."""
    if abs(factor - 1.0) < 1e-3:
        return img
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img)
    rgb = arr[..., :3].astype(np.float32)
    # Luma (Rec. 601) — pull each channel toward/away from grayscale.
    luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
    out = luma + (rgb - luma) * float(factor)
    arr[..., :3] = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def apply_palette(img: Image.Image,
                  palette: list[tuple[int, int, int]],
                  dither: bool) -> Image.Image:
    """Re-quantize an RGBA image's RGB channels to the given palette.

    Alpha is preserved untouched. Fully-transparent pixels are not snapped.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3]
    rgb = Image.fromarray(arr[..., :3], "RGB")
    pal_img = _palette_to_pil(palette)
    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quant = rgb.quantize(palette=pal_img, dither=dither_mode).convert("RGB")
    out = np.array(quant)
    rgba = np.dstack([out, alpha])
    return Image.fromarray(rgba, "RGBA")


def extract_palette_from_image(img: Image.Image,
                               max_colors: int = 32) -> list[tuple[int, int, int]]:
    """Reduce an image to its dominant colors via median-cut."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # Drop fully-transparent pixels by compositing on a neutral background
    # we then ignore — quantize works on RGB.
    rgb = Image.new("RGB", img.size)
    rgb.paste(img, mask=img.split()[3])
    q = rgb.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    used = sorted(set(q.getdata()))
    out: list[tuple[int, int, int]] = []
    for idx in used:
        r, g, b = pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2]
        out.append((int(r), int(g), int(b)))
    return out[:max_colors]


log = logging.getLogger("ppa-sidecar")

app = FastAPI(title="proper-pixel-art sidecar")

# Electron renderer loads from file:// so the Origin header is "null".
# Allow everything — we only bind to localhost anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PixelateRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded source image (PNG/JPEG/etc).")
    num_colors: int = Field(16, ge=1, le=256)
    scale_result: int = Field(1, ge=1, le=32)
    transparent_background: bool = False
    pixel_width: int | None = Field(None, ge=1, le=512)
    clean_edges: bool = True
    outline: bool = False
    outline_thickness: int = Field(1, ge=1, le=16)
    outline_diagonal: bool = True
    dither: bool = False
    saturation: float = Field(1.0, ge=0.0, le=4.0)
    palette: list[tuple[int, int, int]] | None = None
    palette_preset: str | None = None
    key_color: tuple[int, int, int] | None = None
    key_tolerance: int = Field(0, ge=0, le=441)


class ExtractPaletteRequest(BaseModel):
    image_b64: str
    max_colors: int = Field(32, ge=2, le=256)


class ExtractPaletteResponse(BaseModel):
    palette: list[tuple[int, int, int]]


class PixelateResponse(BaseModel):
    image_b64: str
    width: int
    height: int


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/pixelate", response_model=PixelateResponse)
def do_pixelate(req: PixelateRequest) -> PixelateResponse:
    try:
        raw = base64.b64decode(req.image_b64)
        src = Image.open(io.BytesIO(raw))
        # Force load so we get a real exception here, not later.
        src.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    # Resolve any explicit palette (imported list wins over preset name).
    chosen_palette: list[tuple[int, int, int]] | None = None
    if req.palette:
        chosen_palette = [tuple(c) for c in req.palette]
    elif req.palette_preset:
        chosen_palette = PALETTE_PRESETS.get(req.palette_preset)
        if chosen_palette is None:
            raise HTTPException(
                status_code=400,
                detail=f"unknown palette preset: {req.palette_preset}",
            )

    # When the user picked a specific palette we skip pixelate's quantization
    # so we don't double-quantize through an unrelated palette first.
    quant_colors = None if chosen_palette is not None else req.num_colors

    try:
        if req.pixel_width is not None:
            # Explicit pixel width: bypass mesh detection so the output is a
            # true N-to-1 downsample of the source (pixel_width=1 -> input dims).
            result = block_pixelate(
                src, pixel_width=req.pixel_width, num_colors=quant_colors,
            )
            if req.transparent_background:
                result = ppa_colors.make_background_transparent(result)
            if req.scale_result and req.scale_result > 1:
                result = ppa_utils.scale_img(result, int(req.scale_result))
        else:
            result = pixelate(
                src,
                num_colors=quant_colors,
                scale_result=req.scale_result,
                transparent_background=req.transparent_background,
            )
    except Exception as exc:
        log.error("pixelate failed: %s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"pixelate failed: {exc}") from exc

    # Saturation adjustment runs before palette mapping so quantization
    # snaps to the desaturated/boosted colors rather than the originals.
    if abs(req.saturation - 1.0) > 1e-3:
        try:
            result = adjust_saturation(result, req.saturation)
        except Exception as exc:
            log.warning("saturation adjust failed: %s", exc)

    # Apply explicit palette + optional dithering after pixelation.
    if chosen_palette is not None or req.dither:
        try:
            pal_for_apply = chosen_palette
            if pal_for_apply is None:
                # Dithering with no explicit palette: derive one from the
                # already-pixelated result so we still match its colors.
                pal_for_apply = extract_palette_from_image(
                    result, max_colors=max(2, req.num_colors),
                )
            result = apply_palette(result, pal_for_apply, dither=req.dither)
        except Exception as exc:
            log.warning("palette apply failed: %s", exc)

    # Color-key mask: knock out output pixels close to a chosen background
    # color. Done after palette mapping so the comparison is against the final
    # quantized colors, and before outline so the outline traces the new edge.
    if req.key_color is not None:
        try:
            if result.mode != "RGBA":
                result = result.convert("RGBA")
            arr = np.array(result)
            kr, kg, kb = (int(c) for c in req.key_color)
            rgb = arr[..., :3].astype(np.int16)
            dist = np.sqrt(
                (rgb[..., 0] - kr) ** 2
                + (rgb[..., 1] - kg) ** 2
                + (rgb[..., 2] - kb) ** 2
            )
            mask = (dist <= req.key_tolerance) & (arr[..., 3] > 0)
            if mask.any():
                arr[mask] = (0, 0, 0, 0)
                result = Image.fromarray(arr, "RGBA")
        except Exception as exc:
            log.warning("color-key mask failed: %s", exc)

    if req.clean_edges:
        try:
            result = remove_edge_speckles(result)
        except Exception as exc:
            log.warning("clean_edges failed, returning raw result: %s", exc)

    if req.outline:
        try:
            # Scale outline thickness so it's measured in source pixels, not
            # post-zoom pixels — one "pixel" of outline = one art pixel.
            t = req.outline_thickness * max(1, req.scale_result)
            result = add_outline(result, thickness=t, diagonal=req.outline_diagonal)
        except Exception as exc:
            log.warning("outline failed, returning raw result: %s", exc)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return PixelateResponse(
        image_b64=base64.b64encode(buf.getvalue()).decode("ascii"),
        width=result.width,
        height=result.height,
    )


@app.post("/extract_palette", response_model=ExtractPaletteResponse)
def do_extract_palette(req: ExtractPaletteRequest) -> ExtractPaletteResponse:
    try:
        raw = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    try:
        palette = extract_palette_from_image(img, max_colors=req.max_colors)
    except Exception as exc:
        log.error("extract_palette failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"extract failed: {exc}") from exc

    return ExtractPaletteResponse(palette=palette)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[ppa-sidecar] %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
