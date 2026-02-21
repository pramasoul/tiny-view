# Tiny Images Viewer

Interactive GPU-accelerated viewer for the [80 Million Tiny Images](https://groups.csail.mit.edu/vision/TinyImages/) dataset, displaying all 79.3 million 32x32 images on a single Hilbert-curve canvas.

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **Hilbert-curve layout** — all 79M images mapped onto a 16384x16384 grid using a space-filling Hilbert curve, so visually similar images cluster together
- **Seamless zoom** — scroll to zoom from the full mosaic down to individual 32x32 pixels; detail textures load on demand
- **Metadata overlay** — hover over any image to see its keyword, filename, source URL, crawl date, and search engine
- **Keyword search** — press `/` to search by keyword; matching images highlighted on the grid
- **Link checking** — click images to verify source URLs; colored dots show HTTP status (green = live, red = dead, etc.)
- **Hilbert crawl** — press `C` to automatically crawl and check images along the Hilbert curve
- **Image fetching** — live source URLs are fetched at higher resolution and cached in tar shards
- **Dim mode** — press `D` to dim all images except successfully fetched ones

## Controls

| Key | Action |
|-----|--------|
| Scroll | Zoom (toward cursor) |
| Drag | Pan |
| `F` | Toggle fullscreen |
| `M` | Toggle metadata overlay |
| `H` | Toggle help overlay |
| `/` | Keyword search |
| Click | Check source URL |
| `L` | Toggle link-check dots / fetched overlays |
| `C` | Toggle Hilbert crawl from hovered image |
| `D` | Dim mode (show only fetched images) |
| `R` | Reload OK images from cache |
| Home | Reset view |
| `Q` / Escape | Quit |

## Requirements

- Linux (x86_64)
- OpenGL 3.3+
- [pixi](https://pixi.sh) package manager

### Data files (not included)

Place these in the `data/` directory (or use `--data-dir`):

| File | Size | Description |
|------|------|-------------|
| `tiny_images.bin` | ~227 GB | Raw image data (79.3M x 32x32x3 bytes) |
| `tiny_metadata.bin` | ~57 GB | Per-image metadata records (768 bytes each) |

See [METADATA.md](METADATA.md) for the metadata record format.

## Setup

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install

# Run the viewer
pixi run python viewer.py
```

To use a custom data directory:

```bash
pixi run python viewer.py --data-dir /path/to/data
```

## Caches

The viewer generates several cache files in the data directory on first run:

- `avg_colors.npy` — per-image average RGB values
- `hilbert_avg_grid.npy` — 16384x16384 average-color mosaic
- `link_status.bin` — per-image URL check results
- `fetched/` — tar shards of fetched higher-resolution images + SQLite index

## License

[MIT](LICENSE)
