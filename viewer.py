#!/usr/bin/env python3
"""Interactive viewer for the 80 Million Tiny Images dataset.

Controls:
  Scroll     zoom (toward cursor)
  Drag       pan
  F          toggle fullscreen
  M          toggle metadata overlay for hovered image
  Escape     windowed (if fullscreen) / quit
  Q          quit
  Home       reset view
  /          search keywords (type to search, Esc/Enter to cancel)
  Click      check source URL (shows colored dot)
  H          toggle help overlay
  L          toggle link-check dots and fetched overlays
  C          toggle curve crawl from hovered image
  D          dim all except fetched images (find live sources)
  R          reload ok images from cache (or network fallback)
  P          toggle hi-res preview popup
"""

import argparse
import math, os, signal, sqlite3, sys, tarfile, time, threading
import concurrent.futures
import io
import socket
import urllib.request
import urllib.error
import numpy as np
import glfw
import moderngl
from PIL import Image, ImageDraw, ImageFont

# ── Dataset constants ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_SCRIPT_DIR, 'data')

NUM_IMAGES = 79302017
IMG = 32
CH = 3
BPI = IMG * IMG * CH
META_REC = 768          # bytes per metadata record
META_KW_LEN = 80        # keyword field width
ORDER = 14
SIDE = 1 << ORDER          # 16384

GIL_W = 12041
GIL_H = 6586
GIL_CELLS = GIL_W * GIL_H   # 79,302,026


# ── Hilbert curve (numpy-vectorized) ─────────────────────────────────
def hilbert_d2xy(order, d):
    """Convert Hilbert index *d* → (x, y) on a 2^order grid (vectorized)."""
    d = np.asarray(d, dtype=np.int64)
    x = np.zeros_like(d)
    y = np.zeros_like(d)
    t = d.copy()
    s = 1
    while s < (1 << order):
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1
        # rotate quadrant
        swap = (ry == 0)
        flip = swap & (rx == 1)
        x_tmp = x.copy()
        # swap x,y where ry==0
        np.copyto(x, y, where=swap)
        np.copyto(y, x_tmp, where=swap)
        # flip where ry==0 and rx==1
        np.copyto(x, s - 1 - x, where=flip)
        np.copyto(y, s - 1 - y, where=flip)
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return x, y


def hilbert_xy2d(order, x, y):
    """Convert (x, y) → Hilbert index on a 2^order grid (vectorized)."""
    x = np.array(x, dtype=np.int64, copy=True)
    y = np.array(y, dtype=np.int64, copy=True)
    d = np.zeros_like(x)
    s = (1 << order) >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rotate
        swap = (ry == 0)
        flip = swap & (rx == 1)
        x_tmp = x.copy()
        np.copyto(x, y, where=swap)
        np.copyto(y, x_tmp, where=swap)
        np.copyto(x, s - 1 - x, where=flip)
        np.copyto(y, s - 1 - y, where=flip)
        s >>= 1
    return d

# ── Gilbert curve generator (Cerveny's algorithm, BSD-2-Clause) ──────
def _gilbert2d_gen(x, y, ax, ay, bx, by):
    """Recursive generator yielding (col, row) along a Gilbert curve.

    Adapted from Jakub Cerveny's ``gilbert`` reference implementation
    (BSD-2-Clause).  Recursion depth is O(log2(max(W,H))) ≈ 14.
    """
    w = abs(ax + ay)
    h = abs(bx + by)

    dax = 1 if ax > 0 else (-1 if ax < 0 else 0)
    day = 1 if ay > 0 else (-1 if ay < 0 else 0)
    dbx = 1 if bx > 0 else (-1 if bx < 0 else 0)
    dby = 1 if by > 0 else (-1 if by < 0 else 0)

    if h == 1:
        for _ in range(w):
            yield (x, y)
            x += dax
            y += day
        return

    if w == 1:
        for _ in range(h):
            yield (x, y)
            x += dbx
            y += dby
        return

    ax2 = ax // 2
    ay2 = ay // 2
    bx2 = bx // 2
    by2 = by // 2

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 & 1) and (w > 2):
            ax2 += dax
            ay2 += day
        yield from _gilbert2d_gen(x, y, ax2, ay2, bx, by)
        yield from _gilbert2d_gen(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
    else:
        if (h2 & 1) and (h > 2):
            bx2 += dbx
            by2 += dby
        yield from _gilbert2d_gen(x, y, bx2, by2, ax2, ay2)
        yield from _gilbert2d_gen(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from _gilbert2d_gen(x + (ax - dax) + (bx2 - dbx),
                                  y + (ay - day) + (by2 - dby),
                                  -bx2, -by2, -(ax - ax2), -(ay - ay2))


MIN_ZOOM = -18.0  # whole grid → ~1 px (Powers of Ten)
MAX_ZOOM =  5.0   # 32 display-px per image-px
TITLE_ZOOM = -2.0 # show keyword in title bar at this zoom or closer
DETAIL_MIN_PPI = 4 # show detail textures when images are ≥ this many px on screen
META_OFFSET = (100, 100)  # overlay offset from cursor (right, down) in pixels

LINK_TIMEOUT = 10
CRAWL_STOP_RUN = 1024  # stop crawl direction after N consecutive already-checked
DOT_COLORS = {
    'pending':   (1.0, 0.6, 0.0),
    'ok':        (0.0, 0.9, 0.0),
    'moved':     (1.0, 1.0, 0.0),
    'not_found': (1.0, 0.0, 0.0),
    'error':     (0.7, 0.0, 0.0),
    'dns':       (0.8, 0.0, 0.8),
    'no_url':    (0.5, 0.5, 0.5),
    'not_image': (0.4, 0.4, 0.7),
}
# Vectorized color LUT indexed by STATUS_CODES value (0 = unchecked placeholder)
STATUS_COLORS_LUT = np.array([
    [0.05, 0.05, 0.05],  # 0: unchecked
    [1.0,  0.6,  0.0 ],  # 1: pending
    [0.0,  0.9,  0.0 ],  # 2: ok
    [1.0,  1.0,  0.0 ],  # 3: moved
    [1.0,  0.0,  0.0 ],  # 4: not_found
    [0.7,  0.0,  0.0 ],  # 5: error
    [0.8,  0.0,  0.8 ],  # 6: dns
    [0.5,  0.5,  0.5 ],  # 7: no_url
    [0.4,  0.4,  0.7 ],  # 8: not_image
], dtype='f4')
STATUS_CODES = {
    'pending': 1, 'ok': 2, 'moved': 3, 'not_found': 4,
    'error': 5, 'dns': 6, 'no_url': 7, 'not_image': 8,
}
STATUS_NAMES = {v: k for k, v in STATUS_CODES.items()}
FETCH_MAX_SIZE = 512
FETCH_SHARD_MAX = 1 << 31             # 2 GB per tar shard
FETCH_MIN_PPI = 16

# Max detail texture dimension (pixels).  16384 is safe on any modern GPU.
MAX_TEX = 16384

# How much larger than the viewport the detail region should be (multiplier).
# 3.0 means 3× the viewport in each dimension — plenty of margin for panning.
DETAIL_MARGIN = 3.0

# ── Shaders ──────────────────────────────────────────────────────────
VERT = """
#version 330
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }
"""

FRAG = """
#version 330
uniform vec2  u_center;      // view centre in grid coords
uniform float u_ppi;         // display pixels per image
uniform vec2  u_screen;      // viewport size
uniform vec2  u_grid;        // (SIDE, SIDE)

uniform sampler2D u_avg;     // average-colour texture  (SIDE x SIDE)
uniform sampler2D u_detail;  // detail texture           (vis_w*32 x vis_h*32)
uniform vec2  u_det_org;     // grid origin of detail region
uniform vec2  u_det_sz;      // grid extent of detail region (images)
uniform int   u_has_det;     // 1 when detail texture is valid
uniform float u_dim;         // 0.0 = full brightness, >0 darkens output

out vec4 frag;

// ── cubic B-spline weights (smooth, practical for GPU) ──
vec4 cubic(float t) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - t;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) / 6.0;
}

vec4 bicubic(sampler2D tex, vec2 uv, vec2 sz) {
    vec2 inv = 1.0 / sz;
    vec2 tc  = uv * sz - 0.5;
    vec2 f   = fract(tc);
    tc -= f;

    vec4 xc = cubic(f.x);
    vec4 yc = cubic(f.y);

    vec4 c = tc.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xc.x + xc.y, xc.z + xc.w,
                  yc.x + yc.y, yc.z + yc.w);
    vec4 o = c + vec4(xc.y, xc.w, yc.y, yc.w) / s;
    o *= inv.xxyy;

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(mix(texture(tex, o.yw), texture(tex, o.yz), sy),
               mix(texture(tex, o.xw), texture(tex, o.xz), sy), sx);
}

void main() {
    // screen-pixel → grid coordinate  (y flipped so grid-y↓ = screen-y↓)
    vec2 sp = gl_FragCoord.xy;
    sp.y = u_screen.y - sp.y;
    vec2 gp = u_center + (sp - u_screen * 0.5) / u_ppi;

    // bounds
    if (gp.x < 0.0 || gp.y < 0.0 ||
        gp.x >= u_grid.x || gp.y >= u_grid.y) {
        frag = vec4(0.05, 0.05, 0.05, 1.0);
        return;
    }
    ivec2 cell = ivec2(floor(gp));

    // ── detail path ──
    if (u_has_det == 1) {
        vec2 local = gp - u_det_org;
        vec2 uv    = local / u_det_sz;
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
            if (u_ppi > 32.0) {
                frag = bicubic(u_detail, uv, u_det_sz * 32.0);
            } else {
                frag = texture(u_detail, uv);
            }
            frag.rgb *= (1.0 - u_dim);
            return;
        }
    }

    // ── average-colour path ──
    vec2 uv = (vec2(cell) + 0.5) / u_grid;
    frag = texture(u_avg, uv);
    frag.rgb *= (1.0 - u_dim);
}
"""

OVERLAY_VERT = """
#version 330
in vec2 pos;          // unit quad [0,1]
uniform vec4 u_rect;  // (x0, y0, x1, y1) in NDC
out vec2 v_uv;
void main() {
    v_uv = pos;
    vec2 p = mix(u_rect.xy, u_rect.zw, pos);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

OVERLAY_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 frag;
void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    frag = texture(u_tex, uv);
}
"""

DOT_VERT = """
#version 330
in vec2 grid_pos;
in vec3 color;
uniform vec2  u_center;
uniform float u_ppi;
uniform vec2  u_screen;
uniform int   u_square;
out vec3 v_color;
void main() {
    vec2 center = grid_pos + 0.5;
    vec2 sp = (center - u_center) * u_ppi + u_screen * 0.5;
    sp.y = u_screen.y - sp.y;
    vec2 ndc = sp / u_screen * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_square == 1 ? clamp(u_ppi, 1.0, 64.0)
                                 : clamp(u_ppi * 0.4, 6.0, 32.0);
    v_color = color;
}
"""

DOT_FRAG = """
#version 330
in vec3 v_color;
uniform int u_square;
out vec4 frag;
void main() {
    if (u_square == 0) {
        vec2 c = gl_PointCoord - 0.5;
        if (dot(c, c) > 0.25) discard;
    }
    frag = vec4(v_color, 1.0);
}
"""


# ── Gilbert table builder + cache ────────────────────────────────────
def _build_gilbert_tables(w, h, cache_dir):
    """Build or load Gilbert d2xy / xy2d lookup tables.

    Returns (d2xy_x, d2xy_y, xy2d) — int32 arrays.
    First run iterates the generator (~1-2 min for 79M cells), then caches
    as .npy files.  Subsequent runs mmap the cache (~instant).
    """
    n = w * h
    d2xy_path = os.path.join(cache_dir, f'gilbert_{w}x{h}_d2xy.npy')
    xy2d_path = os.path.join(cache_dir, f'gilbert_{w}x{h}_xy2d.npy')

    if os.path.exists(d2xy_path) and os.path.exists(xy2d_path):
        d2xy = np.load(d2xy_path, mmap_mode='r')
        d2xy_x = d2xy[:, 0]
        d2xy_y = d2xy[:, 1]
        xy2d = np.load(xy2d_path, mmap_mode='r')
        return d2xy_x, d2xy_y, xy2d

    print(f"Building Gilbert {w}x{h} tables (one-time, ~1-2 min) …")
    d2xy = np.empty((n, 2), dtype=np.int32)
    t0 = time.time()
    for i, (cx, cy) in enumerate(_gilbert2d_gen(0, 0, w, 0, 0, h)):
        d2xy[i, 0] = cx
        d2xy[i, 1] = cy
        if i % 5_000_000 == 0 and i > 0:
            elapsed = time.time() - t0
            pct = 100 * i / n
            eta = elapsed / pct * (100 - pct) if pct > 0 else 0
            print(f"\r  {pct:5.1f}%  ETA {eta:.0f}s", end='', flush=True)
    print(f"\r  100.0%  ({time.time() - t0:.1f}s)    ")

    # verify adjacency (Manhattan distance 1 between consecutive steps)
    dx = np.abs(np.diff(d2xy[:, 0]))
    dy = np.abs(np.diff(d2xy[:, 1]))
    dist = dx + dy
    bad = np.count_nonzero(dist != 1)
    if bad:
        print(f"  WARNING: {bad} non-adjacent steps in Gilbert curve")

    # build inverse table
    xy2d = np.full((h, w), -1, dtype=np.int32)
    xy2d[d2xy[:, 1], d2xy[:, 0]] = np.arange(n, dtype=np.int32)

    np.save(d2xy_path, d2xy)
    np.save(xy2d_path, xy2d)
    print(f"  Saved to {d2xy_path} and {xy2d_path}")
    return d2xy[:, 0], d2xy[:, 1], xy2d


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


# ── Hilbert avg-grid cache ────────────────────────────────────────────
def _scatter_avg_to_grid(avg):
    """Hilbert-scatter per-image avg colours into a SIDE×SIDE×3 grid."""
    grid = np.full((SIDE, SIDE, 3), 13, dtype='uint8')
    CHUNK = 4_000_000
    for i in range(0, NUM_IMAGES, CHUNK):
        e = min(i + CHUNK, NUM_IMAGES)
        hx, hy = hilbert_d2xy(ORDER, np.arange(i, e, dtype=np.int64))
        grid[hy, hx] = avg[i:e]
    return grid


def load_or_build_grid(data, grid_cache, avg_cache):
    """Load the SIDE×SIDE×3 Hilbert avg-colour grid, building it if needed."""
    if os.path.exists(grid_cache):
        print("Loading cached Hilbert avg grid …")
        return np.load(grid_cache)

    # fast path: build from per-image avg cache (no dataset scan)
    if os.path.exists(avg_cache):
        print("Building Hilbert avg grid from avg cache …")
        avg = np.load(avg_cache)
        grid = _scatter_avg_to_grid(avg)
        np.save(grid_cache, grid)
        print(f"Saved to {grid_cache}")
        return grid

    # slow path: compute averages from raw dataset
    print("Computing average colours (one-time) …")
    avg = np.empty((NUM_IMAGES, 3), dtype='uint8')
    CHUNK = 200_000
    t0 = time.time()
    for i in range(0, NUM_IMAGES, CHUNK):
        e = min(i + CHUNK, NUM_IMAGES)
        avg[i:e] = data[i:e].mean(axis=(2, 3)).astype('uint8')
        elapsed = time.time() - t0
        pct = 100 * e / NUM_IMAGES
        eta = elapsed / pct * (100 - pct) if pct > 0 else 0
        print(f"\r  {pct:5.1f}%  ETA {eta:.0f}s", end='', flush=True)
    print()
    print("Scattering into Hilbert grid …")
    grid = _scatter_avg_to_grid(avg)
    np.save(grid_cache, grid)
    print(f"Saved to {grid_cache}")
    return grid


# ── Gilbert avg-grid ─────────────────────────────────────────────────
def _scatter_avg_to_grid_gilbert(avg, d2xy_x, d2xy_y, w, h):
    """Scatter per-image avg colours into a w×h×3 grid via Gilbert tables."""
    grid = np.full((h, w, 3), 13, dtype='uint8')
    n = len(d2xy_x)
    grid[d2xy_y[:n], d2xy_x[:n]] = avg[:n]
    return grid


def load_or_build_grid_gilbert(data, grid_cache, avg_cache, d2xy_x, d2xy_y, w, h):
    """Load the w×h×3 Gilbert avg-colour grid, building it if needed."""
    if os.path.exists(grid_cache):
        print("Loading cached Gilbert avg grid …")
        return np.load(grid_cache)

    if os.path.exists(avg_cache):
        print("Building Gilbert avg grid from avg cache …")
        avg = np.load(avg_cache)
    else:
        print("Computing average colours (one-time) …")
        avg = np.empty((NUM_IMAGES, 3), dtype='uint8')
        CHUNK = 200_000
        t0 = time.time()
        for i in range(0, NUM_IMAGES, CHUNK):
            e = min(i + CHUNK, NUM_IMAGES)
            avg[i:e] = data[i:e].mean(axis=(2, 3)).astype('uint8')
            elapsed = time.time() - t0
            pct = 100 * e / NUM_IMAGES
            eta = elapsed / pct * (100 - pct) if pct > 0 else 0
            print(f"\r  {pct:5.1f}%  ETA {eta:.0f}s", end='', flush=True)
        print()
        np.save(avg_cache, avg)

    grid = _scatter_avg_to_grid_gilbert(avg, d2xy_x, d2xy_y, w, h)
    np.save(grid_cache, grid)
    print(f"Saved to {grid_cache}")
    return grid


# ── Viewer ────────────────────────────────────────────────────────────
class Viewer:
    def _fit_zoom(self, w, h):
        """Zoom level that fits the whole grid in 90% of the shorter axis."""
        ppi = 0.9 * min(w / self._grid_w, h / self._grid_h)
        return math.log2(max(ppi, 2 ** (MIN_ZOOM + 5))) - 5

    def _acquire_lock(self):
        """Prevent multiple instances from corrupting tar shards."""
        if os.path.exists(self._lock_path):
            try:
                old_pid = int(open(self._lock_path).read().strip())
                os.kill(old_pid, 0)   # check if alive (signal 0 = no-op)
                print(f"Another viewer is running (PID {old_pid}).\n"
                      f"If this is wrong, delete {self._lock_path}",
                      file=sys.stderr, flush=True)
                sys.exit(1)
            except (ValueError, ProcessLookupError):
                pass   # stale lock from a crash — take over
            except PermissionError:
                pass   # process exists but owned by another user — unlikely
        with open(self._lock_path, 'w') as f:
            f.write(str(os.getpid()))

    def _release_lock(self):
        try:
            if (os.path.exists(self._lock_path)
                    and open(self._lock_path).read().strip() == str(os.getpid())):
                os.remove(self._lock_path)
        except OSError:
            pass

    def __init__(self, data_dir=None, verbose=False, quiet=False, layout='hilbert'):
        self._verbose = verbose
        self._quiet = quiet
        self._layout = layout
        self._init_paths(data_dir)
        self._init_layout()
        self._init_view()
        self._init_search()
        self._init_window()        # GLFW + moderngl + shaders + avg texture
        self._init_overlays()      # metadata/help overlay shader + state
        self._init_link_status()   # mmap, dict, thread pool, dot buffer
        self._init_shard_cache()   # fetch_dir, sqlite, tar shards, fi_* arrays
        self._init_crawl()         # crawl threads, status line state
        self._log(f"{self._layout.title()} grid {self._grid_w}×{self._grid_h} = {NUM_IMAGES:,} images")
        self._log("Scroll=zoom  Drag=pan  Click=check  C=crawl  D=dim  R=reload  P=preview  F=full  M=meta  H=help  L=dots  /=search  Esc/Q=quit")

    def _log(self, msg):
        if not self._quiet:
            print(msg, flush=True)

    # ── init helpers ─────────────────────────────────────────────────

    def _init_paths(self, data_dir):
        dd = data_dir or DATA_DIR
        self._filename   = os.path.join(dd, 'tiny_images.bin')
        self._metafile   = os.path.join(dd, 'tiny_metadata.bin')
        self._avg_cache  = os.path.join(dd, 'avg_colors.npy')
        self._grid_cache = os.path.join(dd, 'hilbert_avg_grid.npy')
        self._link_cache = os.path.join(dd, 'link_status.bin')
        self._fetch_dir  = os.path.join(dd, 'fetched')

    def _init_layout(self):
        dd = os.path.dirname(self._filename)
        if self._layout == 'gilbert':
            d2xy_x, d2xy_y, xy2d = _build_gilbert_tables(GIL_W, GIL_H, dd)
            self._grid_w, self._grid_h = GIL_W, GIL_H
            self._gil_d2xy_x = d2xy_x
            self._gil_d2xy_y = d2xy_y
            self._gil_xy2d = xy2d
            self._grid_cache = os.path.join(dd, f'gilbert_{GIL_W}x{GIL_H}_avg_grid.npy')
        else:
            self._grid_w, self._grid_h = SIDE, SIDE

    def _d2xy(self, indices):
        """Convert linear indices → (x_arr, y_arr) using the active layout."""
        indices = np.asarray(indices, dtype=np.int64)
        if self._layout == 'gilbert':
            idx = np.clip(indices, 0, GIL_CELLS - 1)
            return (self._gil_d2xy_x[idx].astype(np.int64),
                    self._gil_d2xy_y[idx].astype(np.int64))
        return hilbert_d2xy(ORDER, indices)

    def _xy2d(self, x, y):
        """Convert (x, y) → linear indices using the active layout."""
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        if self._layout == 'gilbert':
            return self._gil_xy2d[y, x].astype(np.int64)
        return hilbert_xy2d(ORDER, x, y)

    def _init_view(self):
        self.w, self.h = 1920, 1080
        self.cx = self._grid_w / 2.0
        self.cy = self._grid_h / 2.0
        self.zoom = self._fit_zoom(self.w, self.h)
        self.dragging = False
        self.mx = self.my = 0.0

        # detail texture cache
        self._det_key = None
        self._det_tex = None
        self._dirty = True

        # background detail builder
        self._bg_thread = None
        self._bg_cancel = threading.Event()
        self._bg_lock = threading.Lock()
        self._bg_results = []
        self._det_build_vp = None

        # fullscreen state
        self._fullscreen = False
        self._windowed_pos = (100, 100)
        self._windowed_size = (self.w, self.h)

    def _init_search(self):
        self._search = None

    def _init_window(self):
        if not glfw.init():
            sys.exit("glfw.init() failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        self.win = glfw.create_window(self.w, self.h, "Tiny Images", None, None)
        if not self.win:
            glfw.terminate()
            sys.exit("window creation failed")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)

        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_mouse_button_callback(self.win, self._on_button)
        glfw.set_cursor_pos_callback(self.win, self._on_cursor)
        glfw.set_framebuffer_size_callback(self.win, self._on_resize)
        glfw.set_key_callback(self.win, self._on_key)
        glfw.set_char_callback(self.win, self._on_char)

        self.ctx = moderngl.create_context()

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.ctx.buffer(quad), '2f', 'pos')]
        )

        self._log(f"Memory-mapping {self._filename} …")
        self.data = np.memmap(self._filename, dtype='uint8', mode='r',
                              shape=(NUM_IMAGES, CH, IMG, IMG))

        if self._layout == 'gilbert':
            avg_grid = load_or_build_grid_gilbert(
                self.data, self._grid_cache, self._avg_cache,
                self._gil_d2xy_x[:NUM_IMAGES], self._gil_d2xy_y[:NUM_IMAGES],
                self._grid_w, self._grid_h)
        else:
            avg_grid = load_or_build_grid(self.data, self._grid_cache,
                                          self._avg_cache)
        self.avg_tex = self.ctx.texture((self._grid_w, self._grid_h), 3, avg_grid.tobytes())
        del avg_grid
        self.avg_tex.build_mipmaps()
        self.avg_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        # 1×1 dummy for the detail slot when unused
        self._dummy = self.ctx.texture((1, 1), 3, b'\x00\x00\x00')

        self.meta = np.memmap(self._metafile, dtype='uint8', mode='r',
                              shape=(NUM_IMAGES, META_REC))
        self._hover_idx = -1

    def _init_overlays(self):
        self._show_meta = False
        self._show_help = False
        self._meta_tex = None
        self._meta_idx = -1
        self._meta_tw = 0
        self._meta_th = 0
        self._help_tex = None
        self._help_tw = 0
        self._help_th = 0
        self._show_preview = False
        self._preview_tex = None
        self._preview_idx = -1
        self._preview_tw = 0
        self._preview_th = 0

        self._overlay_prog = self.ctx.program(
            vertex_shader=OVERLAY_VERT, fragment_shader=OVERLAY_FRAG)
        overlay_quad = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype='f4')
        self._overlay_vao = self.ctx.vertex_array(
            self._overlay_prog,
            [(self.ctx.buffer(overlay_quad), '2f', 'pos')])

    def _init_link_status(self):
        self._link_lock = threading.Lock()
        self._link_pool = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        self._link_dirty = False
        self._click_sx = 0.0
        self._click_sy = 0.0
        self._show_dots = True

        # persistent link status (mmap'd byte array)
        if not os.path.exists(self._link_cache):
            fp = np.memmap(self._link_cache, dtype='uint8', mode='w+',
                           shape=(NUM_IMAGES,))
            del fp
        self._link_status = np.memmap(self._link_cache, dtype='uint8', mode='r+',
                                      shape=(NUM_IMAGES,))
        self._link_checks = {}
        snapshot = np.array(self._link_status)
        checked = np.nonzero(snapshot)[0]
        del snapshot
        for i in checked:
            self._link_checks[int(i)] = STATUS_NAMES.get(
                int(self._link_status[i]), 'error')
        if checked.size:
            self._log(f"Restored {checked.size:,} link checks from cache")

        self._dim_mode = False

        # dot shader + cached dot buffer
        self._dot_prog = self.ctx.program(
            vertex_shader=DOT_VERT, fragment_shader=DOT_FRAG)
        self._dot_pos = {}
        self._dot_checked = np.empty(0, dtype=np.int64)
        self._dot_gx = np.empty(0, dtype='f4')
        self._dot_gy = np.empty(0, dtype='f4')
        self._dot_buf = None
        self._dot_vao = None
        self._dot_n = 0
        self._dot_cap = 0
        self._dot_last_rebuild = 0.0
        self._tex_mgr_last = 0.0
        self._viewport_changed = True
        self._dot_needs_rebuild = False
        self._dot_journal = set()
        self._dot_journal_lock = threading.Lock()
        self._ok_cells_buf = None
        self._ok_cells_vao = None
        self._ok_cells_n = 0
        self._ok_cells_dirty = True
        self._rebuild_dots()

    def _init_shard_cache(self):
        self._lock_path = os.path.join(self._fetch_dir, 'viewer.pid')
        os.makedirs(self._fetch_dir, exist_ok=True)
        self._acquire_lock()

        self._fetch_db = sqlite3.connect(
            os.path.join(self._fetch_dir, 'index.db'), check_same_thread=False)
        self._fetch_db.execute('PRAGMA journal_mode=WAL')
        self._fetch_db.execute('''CREATE TABLE IF NOT EXISTS images (
            idx INTEGER PRIMARY KEY,
            shard TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            offset INTEGER,
            size INTEGER)''')
        try:
            self._fetch_db.execute('ALTER TABLE images ADD COLUMN offset INTEGER')
        except sqlite3.OperationalError:
            pass
        try:
            self._fetch_db.execute('ALTER TABLE images ADD COLUMN size INTEGER')
        except sqlite3.OperationalError:
            pass
        self._fetch_db.commit()
        need_migrate = self._fetch_db.execute(
            'SELECT COUNT(*) FROM images WHERE offset IS NULL').fetchone()[0]
        if need_migrate:
            self._migrate_offsets()
        self._shard_lock = threading.Lock()
        self._shard_dirty = 0
        self._shard_tf = None
        shard_files = sorted(
            f for f in os.listdir(self._fetch_dir)
            if f.startswith('shard_') and f.endswith('.tar'))
        if shard_files:
            self._shard_num = int(shard_files[-1].split('_')[1].split('.')[0])
            if os.path.getsize(os.path.join(self._fetch_dir, shard_files[-1])) >= FETCH_SHARD_MAX:
                self._shard_num += 1
        else:
            self._shard_num = 0
        cached = self._fetch_db.execute('SELECT COUNT(*) FROM images').fetchone()[0]
        if cached:
            self._log(f"Fetched image cache: {cached} images in {len(shard_files)} shard(s)")

        # fetched image overlay (viewport-scoped GPU textures)
        self._fetched_queue = []
        self._fetched_lock = threading.Lock()
        self._fetched_textures = {}
        self._fetched_loading = set()
        all_cached = [r[0] for r in self._fetch_db.execute('SELECT idx FROM images')]
        if all_cached:
            arr = np.array(all_cached, dtype=np.int64)
            gx, gy = self._d2xy(arr)
            self._fi_idx = arr
            self._fi_gx = gx.astype(np.int32)
            self._fi_gy = gy.astype(np.int32)
            self._fi_set = set(int(i) for i in all_cached)
        else:
            self._fi_idx = np.empty(0, dtype=np.int64)
            self._fi_gx = np.empty(0, dtype=np.int32)
            self._fi_gy = np.empty(0, dtype=np.int32)
            self._fi_set = set()

    def _init_crawl(self):
        self._crawl_cancel = threading.Event()
        self._crawl_threads = []
        self._crawl_origin = -1
        self._crawl_count = [0, 0]
        self._status_last = 0.0
        self._status_shown = False

    # ── properties ───────────────────────────────────────────────────

    @property
    def ppi(self):
        """Display pixels per image at current zoom."""
        return 2.0 ** (self.zoom + 5)

    # ── helpers ──────────────────────────────────────────────────────

    def _screen_to_image_idx(self, sx, sy):
        """Return the linear image index under screen coords (sx, sy), or -1."""
        ppi = self.ppi
        gx = self.cx + (sx - self.w / 2) / ppi
        gy = self.cy + (sy - self.h / 2) / ppi
        ix, iy = int(math.floor(gx)), int(math.floor(gy))
        if ix < 0 or iy < 0 or ix >= self._grid_w or iy >= self._grid_h:
            return -1
        idx = int(self._xy2d(ix, iy))
        return idx if 0 <= idx < NUM_IMAGES else -1

    def _keyword(self, idx):
        """Return the space-stripped keyword for image *idx*."""
        raw = bytes(self.meta[idx, :META_KW_LEN])
        return raw.rstrip(b' ').decode('ascii', errors='replace')

    def _hover_text(self, sx, sy):
        """Return hover string for the image under (sx, sy), or ''."""
        if self.zoom < TITLE_ZOOM:
            return ''
        idx = self._screen_to_image_idx(sx, sy)
        if idx < 0:
            return ''
        self._hover_idx = idx
        return f"#{idx:,}  {self._keyword(idx)}"

    def _update_title(self, sx, sy):
        hover = self._hover_text(sx, sy)
        parts = []
        if self._search is not None:
            parts.append(f"/{self._search}")
        if hover:
            parts.append(hover)
        if self._crawl_threads:
            fwd, bwd = self._crawl_count
            parts.append(f"Crawling from #{self._crawl_origin:,}... [+{fwd}/-{bwd}]")
        glfw.set_window_title(self.win, "  |  ".join(parts) if parts else "Tiny Images")

    # ── metadata overlay ─────────────────────────────────────────────

    def _read_meta(self, idx):
        """Parse full metadata record for image *idx* into a dict."""
        raw = bytes(self.meta[idx])
        return {
            'keyword':       raw[  0: 80].decode('ascii', errors='replace').rstrip(),
            'filename':      raw[ 80:160].decode('ascii', errors='replace').rstrip(),
            'image_spec':    raw[175:180].decode('ascii', errors='replace').rstrip(),
            'crawl_date':    raw[180:212].decode('ascii', errors='replace').strip(),
            'search_engine': raw[212:222].decode('ascii', errors='replace').strip(),
            'source_url':    raw[422:720].decode('ascii', errors='replace').strip(),
        }

    def _meta_fixed_width(self):
        """Return a fixed overlay width (pixels) based on max field widths."""
        if not hasattr(self, '_meta_fw'):
            font = ImageFont.load_default(size=16)
            pad = 8
            # longest possible line: "url:      " + 80 chars
            ruler = "url:      " + "W" * 80
            self._meta_fw = font.getbbox(ruler)[2] + pad * 2
        return self._meta_fw

    def _make_overlay_tex(self, img):
        """Convert a PIL RGBA image to a GL texture. Returns (tex, tw, th)."""
        tw, th = img.size
        tex = self.ctx.texture((tw, th), 4, img.tobytes())
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return tex, tw, th

    def _render_meta_texture(self, idx):
        """Render metadata for image *idx* into an RGBA texture."""
        if idx == self._meta_idx and self._meta_tex is not None:
            return
        meta = self._read_meta(idx)
        lines = [
            f"#{idx:,}",
            f"keyword:  {meta['keyword']}",
            f"file:     {meta['filename']}",
            f"spec:     {meta['image_spec']}",
            f"date:     {meta['crawl_date']}",
            f"engine:   {meta['search_engine']}",
            f"url:      {meta['source_url'][:80]}",
        ]
        font = ImageFont.load_default(size=16)
        pad = 8
        line_h = font.getbbox("Ag")[3] + 4
        tw = self._meta_fixed_width()
        th = line_h * len(lines) + pad * 2

        img = Image.new('RGBA', (tw, th), (0, 0, 0, 180))
        draw = ImageDraw.Draw(img)
        y = pad
        for ln in lines:
            draw.text((pad, y), ln, fill=(255, 255, 255, 255), font=font)
            y += line_h

        if self._meta_tex is not None:
            self._meta_tex.release()
        self._meta_tex, self._meta_tw, self._meta_th = self._make_overlay_tex(img)
        self._meta_idx = idx

    def _render_help_texture(self):
        """Render help overlay into an RGBA texture (cached)."""
        if self._help_tex is not None:
            return
        lines = [
            "Scroll     zoom (toward cursor)",
            "Drag       pan",
            "Click      check source URL",
            "F          toggle fullscreen",
            "M          toggle metadata overlay",
            "H          toggle this help",
            "L          toggle link-check dots",
            "D          dim all except fetched images",
            "C          toggle curve crawl from cursor",
            "R          reload ok images from cache/network",
            "P          toggle hi-res preview popup",
            "/          search keywords",
            "Home       reset view",
            "Esc        windowed (if fullscreen) / quit",
            "Q          quit",
        ]
        # dot color legend: (label, R, G, B)
        dot_legend = [
            ("ok",        0, 230, 0),
            ("moved",     255, 255, 0),
            ("pending",   255, 153, 0),
            ("not_found", 255, 0, 0),
            ("error",     179, 0, 0),
            ("dns",       204, 0, 204),
            ("no_url",    128, 128, 128),
            ("not_image", 102, 102, 179),
        ]
        font = ImageFont.load_default(size=16)
        pad = 12
        line_h = font.getbbox("Ag")[3] + 4
        # measure width from longest line (including legend section)
        legend_label_w = max(font.getbbox(lbl)[2] for lbl, *_ in dot_legend)
        dot_r = line_h // 2 - 1
        legend_line_w = dot_r * 2 + 6 + legend_label_w
        max_w = max(max(font.getbbox(ln)[2] for ln in lines), legend_line_w)
        tw = max_w + pad * 2
        # extra space: blank line + header + legend entries
        th = line_h * (len(lines) + 2 + len(dot_legend)) + pad * 2

        img = Image.new('RGBA', (tw, th), (0, 0, 0, 200))
        draw = ImageDraw.Draw(img)
        y = pad
        for ln in lines:
            # highlight the key part
            parts = ln.split(None, 1)
            if len(parts) == 2:
                draw.text((pad, y), parts[0], fill=(120, 220, 255, 255), font=font)
                kw = font.getbbox(parts[0] + " ")[2]
                draw.text((pad + kw, y), parts[1], fill=(255, 255, 255, 255), font=font)
            else:
                draw.text((pad, y), ln, fill=(255, 255, 255, 255), font=font)
            y += line_h

        # dot color legend
        y += line_h  # blank line
        draw.text((pad, y), "Dot colors:", fill=(120, 220, 255, 255), font=font)
        y += line_h
        for label, r, g, b in dot_legend:
            cx = pad + dot_r
            cy = y + line_h // 2
            draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
                         fill=(r, g, b, 255))
            draw.text((pad + dot_r * 2 + 6, y), label,
                      fill=(255, 255, 255, 255), font=font)
            y += line_h

        self._help_tex, self._help_tw, self._help_th = self._make_overlay_tex(img)

    # ── keyword search ───────────────────────────────────────────────

    def _find_prefix(self, prefix):
        """Binary search for first image whose keyword >= prefix."""
        pb = prefix.encode('ascii')
        n = len(pb)
        lo, hi = 0, NUM_IMAGES
        while lo < hi:
            mid = (lo + hi) // 2
            kw = bytes(self.meta[mid, :n])
            if kw < pb:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _jump_to(self, idx):
        """Centre the view on image *idx*."""
        gx, gy = self._d2xy(np.int64(idx))
        gx, gy = int(gx), int(gy)
        self.cx = gx + 0.5
        self.cy = gy + 0.5
        self._dirty = True

    def _update_search(self):
        if self._search is None or not self._search:
            mx, my = glfw.get_cursor_pos(self.win)
            self._update_title(mx, my)
            return
        idx = self._find_prefix(self._search)
        if idx < NUM_IMAGES:
            kw = self._keyword(idx)
            if kw.startswith(self._search):
                self._jump_to(idx)
                mx, my = glfw.get_cursor_pos(self.win)
                self._update_title(mx, my)
                return
        # no match — show search with hover
        mx, my = glfw.get_cursor_pos(self.win)
        hover = self._hover_text(mx, my)
        parts = [f"/{self._search} (no match)"]
        if hover:
            parts.append(hover)
        glfw.set_window_title(self.win, "  |  ".join(parts))

    # ── link checking ────────────────────────────────────────────────

    def _enqueue_link_check(self, idx, verbose=False):
        """Queue a GET check+fetch for the source URL of image *idx*."""
        with self._link_lock:
            if idx in self._link_checks:
                return
            self._link_checks[idx] = 'pending'
        self._dirty = True
        self._link_pool.submit(self._check_link, idx, verbose)

    def _set_link_status(self, idx, status):
        """Update link status in memory, mmap, and dot journal."""
        with self._link_lock:
            self._link_checks[idx] = status
            self._link_status[idx] = STATUS_CODES.get(status, 5)
        with self._dot_journal_lock:
            self._dot_journal.add(idx)
        self._link_dirty = True

    def _check_link(self, idx, verbose=False):
        """GET the source URL, classify it, and fetch the image (runs in thread pool)."""
        loud = (self._verbose or verbose) and not self._quiet
        try:
            # already fetched — just mark ok and queue for display
            cached = self._load_fetched(idx)
            if cached is not None:
                self._set_link_status(idx, 'ok')
                gx, gy = self._d2xy(np.int64(idx))
                with self._fetched_lock:
                    self._fetched_queue.append((idx, int(gx), int(gy), cached))
                return
            meta = self._read_meta(idx)
            url = meta['source_url'].strip('\x00 ')
            if not url:
                self._set_link_status(idx, 'no_url')
                return
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            if loud:
                print(f"  check #{idx} GET {url[:80]}", flush=True)
            req = urllib.request.Request(
                url, headers={'User-Agent': 'TinyImagesViewer/1.0'})
            opener = urllib.request.build_opener(_NoRedirectHandler)
            resp = opener.open(req, timeout=LINK_TIMEOUT)
            code = resp.getcode()
            if not (200 <= code < 300):
                self._set_link_status(idx, 'error')
                return
            ctype = resp.headers.get('Content-Type', '')
            if not ctype.startswith('image/'):
                if loud:
                    print(f"  check #{idx} not image: {ctype}", flush=True)
                self._set_link_status(idx, 'not_image')
                return
            raw = resp.read(16 * 1024 * 1024)  # 16 MB cap
            if loud:
                print(f"  check #{idx} ok {len(raw)} bytes", flush=True)
            img = Image.open(io.BytesIO(raw)).convert('RGB')
            w, h = img.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
            if s > FETCH_MAX_SIZE:
                img = img.resize((FETCH_MAX_SIZE, FETCH_MAX_SIZE), Image.LANCZOS)
            if not self._save_fetched(idx, img):
                self._set_link_status(idx, 'error')
                return
            pixels = np.asarray(img, dtype='uint8').copy()
            if loud:
                print(f"  check #{idx} image {w}x{h} → {pixels.shape}", flush=True)
            self._set_link_status(idx, 'ok')
            gx, gy = self._d2xy(np.int64(idx))
            with self._fetched_lock:
                self._fetched_queue.append((idx, int(gx), int(gy), pixels))
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 307, 308):
                self._set_link_status(idx, 'moved')
            elif e.code == 404:
                self._set_link_status(idx, 'not_found')
            else:
                self._set_link_status(idx, 'error')
        except socket.gaierror:
            self._set_link_status(idx, 'dns')
        except Exception as e:
            if loud:
                print(f"  check #{idx} FAILED: {e}", flush=True)
            self._set_link_status(idx, 'error')

    # ── shard cache ──────────────────────────────────────────────────

    def _migrate_offsets(self):
        """One-time migration: scan tar shards to populate offset/size."""
        rows = self._fetch_db.execute(
            'SELECT DISTINCT shard FROM images WHERE offset IS NULL').fetchall()
        total = 0
        for (shard_name,) in rows:
            shard_path = os.path.join(self._fetch_dir, shard_name)
            if not os.path.exists(shard_path):
                continue
            with tarfile.open(shard_path, 'r') as tf:
                for member in tf.getmembers():
                    name = member.name.replace('.jpg', '')
                    try:
                        idx = int(name)
                    except ValueError:
                        continue
                    self._fetch_db.execute(
                        'UPDATE images SET offset=?, size=? WHERE idx=? AND offset IS NULL',
                        (member.offset_data, member.size, idx))
                    total += 1
        self._fetch_db.commit()
        if total:
            self._log(f"Migrated {total} cache entries with byte offsets")

    def _open_shard(self):
        """Open (or reopen) the current shard tar for appending.

        If the tar is corrupt (e.g. from a crash mid-write), truncate
        to the last complete entry and write a proper end-of-archive
        marker (two 512-byte zero blocks) so tarfile can reopen it.
        """
        if self._shard_tf is not None:
            self._shard_tf.close()
            self._shard_tf = None
        shard = f"shard_{self._shard_num:06d}.tar"
        path = os.path.join(self._fetch_dir, shard)
        try:
            self._shard_tf = tarfile.open(path, 'a')
        except tarfile.ReadError:
            self._repair_shard(path)
            self._shard_tf = tarfile.open(path, 'a')
        return shard

    def _repair_shard(self, path):
        """Truncate a corrupt tar to the last complete entry and re-terminate it."""
        file_size = os.path.getsize(path)
        safe = 0
        with open(path, 'rb') as f:
            while True:
                header = f.read(512)
                if len(header) < 512 or header == b'\x00' * 512:
                    break  # EOF or end-of-archive marker
                try:
                    info = tarfile.TarInfo.frombuf(header, tarfile.ENCODING,
                                                   'surrogateescape')
                except tarfile.HeaderError:
                    break  # corrupt header
                data_blocks = (info.size + 511) // 512
                entry_end = f.tell() + data_blocks * 512
                if entry_end > file_size:
                    break  # data truncated
                f.seek(data_blocks * 512, 1)
                safe = entry_end
        self._log(f"Repairing {path}: truncating to {safe} and re-terminating "
                  f"({file_size - safe} bytes removed)")
        with open(path, 'r+b') as f:
            f.seek(safe)
            f.write(b'\x00' * 1024)  # two 512-byte end-of-archive blocks
            f.truncate()

    def _save_fetched(self, idx, img):
        """Save a fetched PIL image to the current tar shard. Returns True on success."""
        try:
            with self._shard_lock:
                if self._fetch_db.execute(
                        'SELECT 1 FROM images WHERE idx=?', (idx,)).fetchone():
                    return True
                jpeg_buf = io.BytesIO()
                img.save(jpeg_buf, format='JPEG', quality=90)
                jpeg_data = jpeg_buf.getvalue()
                w, h = img.size
                if self._shard_tf is None:
                    shard = self._open_shard()
                else:
                    shard = f"shard_{self._shard_num:06d}.tar"
                info = tarfile.TarInfo(name=f"{idx:010d}.jpg")
                info.size = len(jpeg_data)
                data_offset = self._shard_tf.fileobj.tell() + 512  # header is 512 bytes
                self._shard_tf.addfile(info, io.BytesIO(jpeg_data))
                # Check shard size, rotate if needed
                pos = self._shard_tf.fileobj.tell()
                if pos >= FETCH_SHARD_MAX:
                    self._shard_num += 1
                    shard = self._open_shard()
                self._fetch_db.execute(
                    'INSERT INTO images (idx, shard, width, height, offset, size) VALUES (?,?,?,?,?,?)',
                    (idx, shard, w, h, data_offset, len(jpeg_data)))
                self._shard_dirty += 1
                if self._shard_dirty >= 50:
                    self._fetch_db.commit()
                    self._shard_dirty = 0
                return True
        except Exception as e:
            if self._verbose:
                print(f"  cache save #{idx} FAILED: {e}", flush=True)
            return False

    def _load_fetched(self, idx):
        """Load a fetched image from the tar cache. Returns pixels array or None."""
        row = self._fetch_db.execute(
            'SELECT shard, offset, size FROM images WHERE idx=?', (idx,)).fetchone()
        if row is None:
            return None
        shard_name, offset, size = row
        shard_path = os.path.join(self._fetch_dir, shard_name)
        if offset is None or size is None:
            return None
        try:
            with open(shard_path, 'rb') as f:
                f.seek(offset)
                data = f.read(size)
            img = Image.open(io.BytesIO(data)).convert('RGB')
            return np.asarray(img, dtype='uint8').copy()
        except Exception as e:
            if self._verbose:
                print(f"  cache load #{idx} FAILED: {e}", flush=True)
            return None

    def _reload_one(self, idx):
        """Load one image from cache or network, queue for GPU upload."""
        pixels = self._load_fetched(idx)
        if pixels is None:
            # not in cache — do a full check+fetch
            self._check_link(idx)
            self._fetched_loading.discard(idx)
            return
        gx, gy = self._d2xy(np.int64(idx))
        with self._fetched_lock:
            self._fetched_queue.append((idx, int(gx), int(gy), pixels))

    def _reload_ok_images(self):
        """Reload all 'ok' images from cache (or network fallback)."""
        with self._link_lock:
            ok_indices = [i for i, s in self._link_checks.items()
                          if s == 'ok' and i not in self._fetched_textures]
        if not ok_indices:
            if self._verbose:
                print("No ok images to reload", flush=True)
            return
        if self._verbose:
            print(f"Reloading {len(ok_indices)} ok images …", flush=True)
        for idx in ok_indices:
            self._link_pool.submit(self._reload_one, idx)

    # ── Hilbert crawl ────────────────────────────────────────────────

    def _crawl_worker(self, start_idx, direction):
        """Walk the Hilbert curve from start_idx, direction = +1 or -1."""
        slot = 0 if direction > 0 else 1
        label = "fwd" if direction > 0 else "bwd"
        idx = start_idx
        consec = 0
        while not self._crawl_cancel.is_set():
            idx += direction
            if idx < 0 or idx >= NUM_IMAGES:
                break
            if self._link_status[idx]:
                consec += 1
                if consec >= CRAWL_STOP_RUN:
                    if self._verbose:
                        print(f"  crawl {label} stopped: {CRAWL_STOP_RUN} consecutive already-checked at #{idx:,}", flush=True)
                    break
                continue   # skip already-checked instantly
            consec = 0
            self._enqueue_link_check(idx)
            self._crawl_count[slot] += 1
            self._crawl_cancel.wait(0.05)

    def _start_crawl(self, idx):
        self._stop_crawl()
        self._enqueue_link_check(idx)
        self._crawl_cancel = threading.Event()
        self._crawl_origin = idx
        self._crawl_count = [0, 0]
        fwd = threading.Thread(target=self._crawl_worker, args=(idx, +1), daemon=True)
        bwd = threading.Thread(target=self._crawl_worker, args=(idx, -1), daemon=True)
        self._crawl_threads = [fwd, bwd]
        fwd.start()
        bwd.start()
        if self._verbose:
            print(f"Crawl started from #{idx:,}", flush=True)

    def _stop_crawl(self):
        if self._crawl_threads:
            self._crawl_cancel.set()
            for t in self._crawl_threads:
                t.join(timeout=1.0)
            total = sum(self._crawl_count)
            self._print_status()
            if self._status_shown:
                print(flush=True)  # newline after final status
                self._status_shown = False
            if total and self._verbose:
                print(f"Crawl stopped ({total:,} checked)", flush=True)
            self._crawl_threads = []

    def _print_status(self):
        """Overwrite the current terminal line with aggregate crawl stats."""
        if self._quiet:
            return
        snap = np.array(self._link_status)
        counts = np.bincount(snap, minlength=9)
        # counts[0] = unchecked, 1=pending, 2=ok, 3=moved, 4=not_found,
        #             5=error, 6=dns, 7=no_url, 8=not_image
        total = int(counts[1:].sum())
        pending = int(counts[1])
        parts = []
        if counts[2]: parts.append(f"ok:{counts[2]}")
        if counts[3]: parts.append(f"moved:{counts[3]}")
        if counts[4]: parts.append(f"404:{counts[4]}")
        if counts[5]: parts.append(f"err:{counts[5]}")
        if counts[6]: parts.append(f"dns:{counts[6]}")
        if counts[7]: parts.append(f"no_url:{counts[7]}")
        if counts[8]: parts.append(f"!img:{counts[8]}")
        line = ' '.join(parts)
        line += f" | cached:{len(self._fi_set)} pend:{pending} total:{total}"
        print(f"\r{line}\033[K", end='', flush=True)
        self._status_shown = True

    # ── input callbacks ──────────────────────────────────────────────

    def _on_scroll(self, win, _dx, dy):
        mx, my = glfw.get_cursor_pos(win)
        old_ppi = self.ppi

        gx = self.cx + (mx - self.w / 2) / old_ppi
        gy = self.cy + (my - self.h / 2) / old_ppi

        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom + dy * 0.15))

        new_ppi = self.ppi
        self.cx = gx - (mx - self.w / 2) / new_ppi
        self.cy = gy - (my - self.h / 2) / new_ppi

        self._hover_idx = -1
        self._update_title(mx, my)
        self._dirty = True
        self._viewport_changed = True

    def _on_button(self, win, btn, action, _mods):
        if btn == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.dragging = True
                self.mx, self.my = glfw.get_cursor_pos(win)
                self._click_sx, self._click_sy = self.mx, self.my
            elif action == glfw.RELEASE:
                self.dragging = False
                rx, ry = glfw.get_cursor_pos(win)
                dx, dy = rx - self._click_sx, ry - self._click_sy
                if dx * dx + dy * dy < 9:
                    idx = self._screen_to_image_idx(rx, ry)
                    if idx >= 0:
                        self._enqueue_link_check(idx, verbose=True)

    def _on_cursor(self, win, x, y):
        if self.dragging:
            ppi = self.ppi
            self.cx -= (x - self.mx) / ppi
            self.cy -= (y - self.my) / ppi
            self.mx, self.my = x, y
            self._dirty = True
            self._viewport_changed = True
        self._update_title(x, y)
        if self._show_meta or self._show_help or self._show_preview:
            self._dirty = True

    def _on_char(self, _win, codepoint):
        ch = chr(codepoint)
        if self._search is None:
            if ch == '/':
                self._search = ''
                self._update_search()
            return
        if ch.isprintable():
            self._search += ch.lower()
            self._update_search()

    def _on_resize(self, _win, w, h):
        if w > 0 and h > 0:
            self.w, self.h = w, h
            self.ctx.viewport = (0, 0, w, h)
            self._dirty = True
            self._viewport_changed = True

    def _on_key(self, win, key, _sc, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        # search editing (when search is active)
        if self._search is not None:
            if key == glfw.KEY_BACKSPACE:
                if self._search:
                    self._search = self._search[:-1]
                else:
                    self._search = None
                self._update_search()
                return
            if key in (glfw.KEY_ESCAPE, glfw.KEY_ENTER):
                self._search = None
                self._update_search()
                return
            return  # all other keys go to _on_char for search input
        # normal keys (not searching)
        if key == glfw.KEY_ESCAPE:
            if self._fullscreen:
                self._leave_fullscreen()
            else:
                glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_Q:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_HOME:
            self.cx, self.cy = self._grid_w / 2.0, self._grid_h / 2.0
            self.zoom = self._fit_zoom(self.w, self.h)
            self._dirty = True
        elif key == glfw.KEY_F:
            self._toggle_fullscreen()
        elif key == glfw.KEY_M:
            self._show_meta = not self._show_meta
            if self._show_meta:
                self._show_help = False
            self._dirty = True
        elif key == glfw.KEY_H:
            self._show_help = not self._show_help
            if self._show_help:
                self._show_meta = False
            self._dirty = True
        elif key == glfw.KEY_L:
            self._show_dots = not self._show_dots
            self._dirty = True
        elif key == glfw.KEY_C:
            if self._crawl_threads:
                self._stop_crawl()
            else:
                idx = self._screen_to_image_idx(*glfw.get_cursor_pos(win))
                if idx >= 0:
                    self._start_crawl(idx)
            self._dirty = True
        elif key == glfw.KEY_D:
            self._dim_mode = not self._dim_mode
            self._dirty = True
        elif key == glfw.KEY_R:
            self._reload_ok_images()
            self._dirty = True
        elif key == glfw.KEY_P:
            self._show_preview = not self._show_preview
            if not self._show_preview and self._preview_tex is not None:
                self._preview_tex.release()
                self._preview_tex = None
                self._preview_idx = -1
            self._dirty = True

    # ── fullscreen ───────────────────────────────────────────────────

    def _toggle_fullscreen(self):
        if self._fullscreen:
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        self._windowed_pos = glfw.get_window_pos(self.win)
        self._windowed_size = glfw.get_window_size(self.win)
        mon = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(mon)
        glfw.set_window_monitor(self.win, mon, 0, 0,
                                mode.size.width, mode.size.height,
                                mode.refresh_rate)
        self._fullscreen = True
        self._dirty = True

    def _leave_fullscreen(self):
        x, y = self._windowed_pos
        w, h = self._windowed_size
        glfw.set_window_monitor(self.win, None, x, y, w, h, 0)
        self._fullscreen = False
        self._dirty = True

    # ── detail texture ───────────────────────────────────────────────

    def _viewport_rect(self, margin=0):
        """(gx0, gy0, gx1, gy1) of visible grid cells, optionally expanded by margin."""
        ppi = self.ppi
        hw, hh = self.w / (2 * ppi), self.h / (2 * ppi)
        if margin:
            return (int(self.cx - hw) - margin, int(self.cy - hh) - margin,
                    int(self.cx + hw) + margin + 1, int(self.cy + hh) + 1 + margin)
        return (self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)

    def _detail_feasible(self):
        """Can we have a detail texture at the current zoom?"""
        if self.ppi < DETAIL_MIN_PPI:
            return False
        # Don't build if viewport exceeds max texture size (would just clamp)
        max_imgs = MAX_TEX // IMG
        gx0, gy0, gx1, gy1 = self._viewport_rect()
        return (gx1 - gx0 + 3) <= max_imgs and (gy1 - gy0 + 3) <= max_imgs

    def _detail_covers_viewport(self):
        """Does the cached detail texture fully cover the current viewport?"""
        if not self._det_key or self._det_tex is None:
            return False
        vp = self._viewport_rect()
        cx0, cy0, cx1, cy1 = self._det_key
        return cx0 <= vp[0] and cy0 <= vp[1] and cx1 >= vp[2] and cy1 >= vp[3]

    def _build_region(self, data, gx0, gy0, gx1, gy1, cancel=None):
        """Assemble a pixel buffer for a grid region."""
        vw, vh = gx1 - gx0, gy1 - gy0
        buf = np.full((vh * IMG, vw * IMG, CH), 13, dtype='uint8')

        # grid coords for every cell in the region
        gxs, gys = np.meshgrid(
            np.arange(gx0, gx1, dtype=np.int64),
            np.arange(gy0, gy1, dtype=np.int64))
        gxs = gxs.ravel()
        gys = gys.ravel()

        # curve index for each cell
        file_idx = self._xy2d(gxs, gys)

        # filter valid (Gilbert holes have xy2d == -1)
        valid = (file_idx >= 0) & (file_idx < NUM_IMAGES)
        file_idx = file_idx[valid]
        gxs = gxs[valid]
        gys = gys[valid]

        if len(file_idx) == 0:
            return buf, vw, vh

        # sort by file index for sequential disk access
        sort_order = np.argsort(file_idx)
        file_idx = file_idx[sort_order]
        gxs = gxs[sort_order]
        gys = gys[sort_order]

        # detect contiguous runs
        breaks = np.where(np.diff(file_idx) != 1)[0] + 1
        run_starts = np.concatenate(([0], breaks))
        run_ends = np.concatenate((breaks, [len(file_idx)]))

        # read each contiguous run and scatter into output buffer
        ri = np.arange(IMG)
        for rs, re in zip(run_starts, run_ends):
            if cancel is not None and cancel.is_set():
                return buf, vw, vh
            fi_start = int(file_idx[rs])
            fi_end = int(file_idx[re - 1]) + 1
            chunk = np.array(data[fi_start:fi_end])  # single contiguous read
            # orientation fix: (N, C, H, W) → (N, H, W, C), rot90 CW, BGR→RGB
            chunk = chunk.transpose(0, 2, 3, 1)
            chunk = np.rot90(chunk, k=-1, axes=(1, 2))
            chunk = chunk[:, :, ::-1]
            # vectorized scatter via advanced indexing
            lx = gxs[rs:re] - gx0
            ly = gys[rs:re] - gy0
            row_idx = ly[:, None, None] * IMG + ri[None, :, None]
            col_idx = lx[:, None, None] * IMG + ri[None, None, :]
            buf[row_idx, col_idx] = chunk

        return buf, vw, vh

    def _start_detail_build(self):
        """Kick off background thread: viewport first, then full margins."""
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_cancel.set()
            self._bg_thread.join(timeout=2.0)
        self._bg_cancel = threading.Event()
        cancel = self._bg_cancel

        ppi = self.ppi
        cx, cy = self.cx, self.cy
        w, h = self.w, self.h
        data = self.data
        self._det_build_vp = (cx, cy, ppi, w, h)
        vp1 = self._viewport_rect(margin=1)
        grid_w, grid_h = self._grid_w, self._grid_h
        build_region = self._build_region

        def clamp_region(gx0, gy0, gx1, gy1):
            max_imgs = MAX_TEX // IMG
            gx0 = max(0, gx0); gy0 = max(0, gy0)
            gx1 = min(grid_w, gx1); gy1 = min(grid_h, gy1)
            vw, vh = gx1 - gx0, gy1 - gy0
            if vw > max_imgs:
                ex = vw - max_imgs
                gx0 += ex // 2; gx1 = gx0 + max_imgs
            if vh > max_imgs:
                ex = vh - max_imgs
                gy0 += ex // 2; gy1 = gy0 + max_imgs
            return gx0, gy0, gx1, gy1

        def worker():
            # Phase 1: viewport only (small, fast)
            vgx0, vgy0, vgx1, vgy1 = clamp_region(*vp1)
            if vgx1 > vgx0 and vgy1 > vgy0:
                buf, vw, vh = build_region(
                    data, vgx0, vgy0, vgx1, vgy1, cancel=cancel)
                if cancel.is_set():
                    return
                with self._bg_lock:
                    self._bg_results.append(
                        ((vgx0, vgy0, vgx1, vgy1), buf, vw, vh))

            if cancel.is_set():
                return

            # Phase 2: full margin region
            hw = w / (2 * ppi)
            hh = h / (2 * ppi)
            mw = hw * DETAIL_MARGIN
            mh = hh * DETAIL_MARGIN
            mgx0, mgy0, mgx1, mgy1 = clamp_region(
                int(cx - mw), int(cy - mh),
                int(cx + mw) + 1, int(cy + mh) + 1)
            if mgx1 > mgx0 and mgy1 > mgy0:
                buf, vw, vh = build_region(
                    data, mgx0, mgy0, mgx1, mgy1, cancel=cancel)
                if cancel.is_set():
                    return
                with self._bg_lock:
                    self._bg_results.append(
                        ((mgx0, mgy0, mgx1, mgy1), buf, vw, vh))

        self._bg_thread = threading.Thread(target=worker, daemon=True)
        self._bg_thread.start()

    def _upload_pending_detail(self):
        """Upload the latest completed result from the bg thread."""
        with self._bg_lock:
            results = list(self._bg_results)
            self._bg_results.clear()
        if not results:
            return False

        # take the last (most complete) result
        key, buf, vw, vh = results[-1]
        if self._det_tex is not None:
            self._det_tex.release()
        tex = self.ctx.texture((vw * IMG, vh * IMG), CH, buf.tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._det_tex = tex
        self._det_key = key
        return True

    # ── fetched textures ─────────────────────────────────────────────

    def _upload_pending_fetches(self):
        """Upload queued fetched images to GPU textures."""
        with self._fetched_lock:
            queue = list(self._fetched_queue)
            self._fetched_queue.clear()
        if not queue:
            return False
        new_idx, new_gx, new_gy = [], [], []
        for idx, gx, gy, pixels in queue:
            if idx not in self._fi_set:
                self._fi_set.add(idx)
                new_idx.append(idx)
                new_gx.append(gx)
                new_gy.append(gy)
            self._fetched_loading.discard(idx)
            h, w = pixels.shape[:2]
            tex = self.ctx.texture((w, h), 3, pixels.tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            if idx in self._fetched_textures:
                self._fetched_textures[idx][0].release()
            self._fetched_textures[idx] = (tex, gx, gy)
        if new_idx:
            self._fi_idx = np.concatenate([self._fi_idx,
                                           np.array(new_idx, dtype=np.int64)])
            self._fi_gx = np.concatenate([self._fi_gx,
                                          np.array(new_gx, dtype=np.int32)])
            self._fi_gy = np.concatenate([self._fi_gy,
                                          np.array(new_gy, dtype=np.int32)])
        return True

    def _manage_fetched_textures(self):
        """Evict off-screen textures, load visible cached images."""
        if self.ppi < FETCH_MIN_PPI:
            if self._fetched_textures:
                for tex, _gx, _gy in self._fetched_textures.values():
                    tex.release()
                self._fetched_textures.clear()
                self._dirty = True
            self._fetched_loading.clear()
            return
        gx0, gy0, gx1, gy1 = self._viewport_rect(margin=2)
        # evict off-screen (wider margin to avoid load/evict churn)
        ex0, ey0, ex1, ey1 = self._viewport_rect(margin=16)
        for idx in list(self._fetched_textures):
            _, gx, gy = self._fetched_textures[idx]
            if not (ex0 <= gx <= ex1 and ey0 <= gy <= ey1):
                self._fetched_textures[idx][0].release()
                del self._fetched_textures[idx]
        # load visible cached images that lack textures (vectorized scan)
        submitted = False
        if len(self._fi_idx) > 0:
            vis = ((self._fi_gx >= gx0) & (self._fi_gx <= gx1) &
                   (self._fi_gy >= gy0) & (self._fi_gy <= gy1))
            for idx_i in self._fi_idx[vis]:
                idx = int(idx_i)
                if (idx not in self._fetched_textures
                        and idx not in self._fetched_loading):
                    self._fetched_loading.add(idx)
                    self._link_pool.submit(self._reload_one, idx)
                    submitted = True
        # also retry ok images not in cache (e.g. checked before fetch existed)
        ok_code = STATUS_CODES['ok']
        vgx = np.arange(max(0, gx0), min(self._grid_w, gx1), dtype=np.int64)
        vgy = np.arange(max(0, gy0), min(self._grid_h, gy1), dtype=np.int64)
        if len(vgx) > 0 and len(vgy) > 0:
            mgx, mgy = np.meshgrid(vgx, vgy)
            vidx = self._xy2d(mgx.ravel(), mgy.ravel())
            vidx = vidx[(vidx >= 0) & (vidx < NUM_IMAGES)]
            for i in vidx[self._link_status[vidx] == ok_code]:
                idx = int(i)
                if (idx not in self._fi_set
                        and idx not in self._fetched_textures
                        and idx not in self._fetched_loading):
                    self._fetched_loading.add(idx)
                    self._link_pool.submit(self._reload_one, idx)
                    submitted = True
        if submitted:
            self._dirty = True

    # ── dot buffer ───────────────────────────────────────────────────

    _DOT_STRIDE = 20  # 5 floats × 4 bytes: gx, gy, r, g, b

    def _dot_ensure_capacity(self, needed):
        """Grow the GPU buffer if needed, preserving existing data."""
        if needed <= self._dot_cap:
            return
        new_cap = max(needed, self._dot_cap * 2, 4096)
        new_buf = self.ctx.buffer(reserve=new_cap * self._DOT_STRIDE)
        if self._dot_buf is not None and self._dot_n > 0:
            old_data = self._dot_buf.read(self._dot_n * self._DOT_STRIDE)
            new_buf.write(old_data)
            self._dot_buf.release()
        if self._dot_vao is not None:
            self._dot_vao.release()
        self._dot_buf = new_buf
        self._dot_vao = self.ctx.vertex_array(
            self._dot_prog, [(self._dot_buf, '2f 3f', 'grid_pos', 'color')])
        self._dot_cap = new_cap

    def _rebuild_dots(self):
        """Rebuild dot buffer incrementally using the change journal."""
        with self._dot_journal_lock:
            journal = self._dot_journal
            self._dot_journal = set()
        self._dot_journal_had_updates = bool(journal)

        if self._dot_n == 0 and not self._dot_pos:
            # Cold start: full scan of mmap to find all checked indices
            snapshot = np.array(self._link_status)
            checked = np.nonzero(snapshot)[0].astype(np.int64)
            n = len(checked)
            if n == 0:
                return
            gx, gy = self._d2xy(checked)
            codes = snapshot[checked]
            colors = STATUS_COLORS_LUT[codes]
            vdata = np.empty((n, 5), dtype='f4')
            vdata[:, 0] = gx.astype('f4')
            vdata[:, 1] = gy.astype('f4')
            vdata[:, 2:5] = colors
            self._dot_ensure_capacity(n)
            self._dot_buf.write(vdata.tobytes())
            self._dot_pos = {int(idx): i for i, idx in enumerate(checked)}
            self._dot_checked = checked
            self._dot_gx = gx.astype('f4')
            self._dot_gy = gy.astype('f4')
            self._dot_n = n
            return

        if not journal:
            return

        self._ok_cells_dirty = True
        # Separate new dots from color updates
        new_dots = []
        color_updates = []
        for idx in journal:
            if idx in self._dot_pos:
                color_updates.append(idx)
            else:
                new_dots.append(idx)

        # Color updates: batch compute colors, individual GPU writes
        if color_updates:
            upd_arr = np.array(color_updates, dtype=np.int64)
            codes = self._link_status[upd_arr]
            colors = STATUS_COLORS_LUT[codes].astype('f4')
            for i, idx in enumerate(color_updates):
                slot = self._dot_pos[idx]
                self._dot_buf.write(colors[i].tobytes(),
                                    offset=slot * self._DOT_STRIDE + 8)

        # New dots: compute positions, append to buffer
        if new_dots:
            new_arr = np.array(new_dots, dtype=np.int64)
            ngx, ngy = self._d2xy(new_arr)
            codes = self._link_status[new_arr]
            colors = STATUS_COLORS_LUT[codes]
            n_new = len(new_dots)
            self._dot_ensure_capacity(self._dot_n + n_new)
            vdata = np.empty((n_new, 5), dtype='f4')
            vdata[:, 0] = ngx.astype('f4')
            vdata[:, 1] = ngy.astype('f4')
            vdata[:, 2:5] = colors
            self._dot_buf.write(vdata.tobytes(),
                                offset=self._dot_n * self._DOT_STRIDE)
            for i, idx in enumerate(new_dots):
                self._dot_pos[idx] = self._dot_n + i
            # Extend cached arrays for ok-cells feature
            self._dot_checked = np.concatenate([self._dot_checked, new_arr])
            self._dot_gx = np.concatenate([self._dot_gx, ngx.astype('f4')])
            self._dot_gy = np.concatenate([self._dot_gy, ngy.astype('f4')])
            self._dot_n += n_new

    # ── render ───────────────────────────────────────────────────────

    def _render(self):
        """Render immediately with whatever detail texture is cached."""
        self.ctx.clear(0.05, 0.05, 0.05)
        self._render_base_map()
        self._render_ok_cells()
        self._render_fetched()
        self._render_dots()
        self._render_preview()
        self._render_info_overlay()

    def _render_base_map(self):
        """Draw the avg-colour grid with optional detail texture overlay."""
        det_ok = self._det_tex is not None and self._det_key is not None
        self.avg_tex.use(0)
        if det_ok:
            cx0, cy0, cx1, cy1 = self._det_key
            self._det_tex.use(1)
            self.prog['u_has_det'].value = 1
            self.prog['u_det_org'].value = (float(cx0), float(cy0))
            self.prog['u_det_sz'].value  = (float(cx1 - cx0), float(cy1 - cy0))
        else:
            self._dummy.use(1)
            self.prog['u_has_det'].value = 0
            self.prog['u_det_org'].value = (0.0, 0.0)
            self.prog['u_det_sz'].value  = (1.0, 1.0)

        self.prog['u_avg'].value    = 0
        self.prog['u_detail'].value = 1
        self.prog['u_center'].value = (self.cx, self.cy)
        self.prog['u_ppi'].value    = self.ppi
        self.prog['u_screen'].value = (float(self.w), float(self.h))
        self.prog['u_grid'].value   = (float(self._grid_w), float(self._grid_h))
        self.prog['u_dim'].value    = 0.85 if self._dim_mode else 0.0

        self.vao.render(moderngl.TRIANGLE_STRIP)

    def _render_ok_cells(self):
        """Draw undimmed ok cells when in dim mode and zoomed out past hi-res."""
        if not (self._dim_mode and self.ppi < FETCH_MIN_PPI and self._dot_n > 0):
            return
        if self._ok_cells_dirty:
            self._ok_cells_dirty = False
            ok_mask = self._link_status[self._dot_checked] == STATUS_CODES['ok']
            ok_count = ok_mask.sum()
            if self._ok_cells_buf is not None:
                self._ok_cells_vao.release()
                self._ok_cells_buf.release()
                self._ok_cells_buf = None
                self._ok_cells_vao = None
                self._ok_cells_n = 0
            if ok_count > 0:
                ok_gx = self._dot_gx[ok_mask]
                ok_gy = self._dot_gy[ok_mask]
                ok_arr = self._dot_checked[ok_mask]
                raw_avg = self.data[ok_arr].reshape(ok_count, CH, -1).mean(axis=2)
                vdata = np.empty((ok_count, 5), dtype='f4')
                vdata[:, 0] = ok_gx
                vdata[:, 1] = ok_gy
                vdata[:, 2] = raw_avg[:, 2] / 255.0  # R (was B)
                vdata[:, 3] = raw_avg[:, 1] / 255.0  # G
                vdata[:, 4] = raw_avg[:, 0] / 255.0  # B (was R)
                self._ok_cells_buf = self.ctx.buffer(vdata.tobytes())
                self._ok_cells_vao = self.ctx.vertex_array(
                    self._dot_prog,
                    [(self._ok_cells_buf, '2f 3f', 'grid_pos', 'color')])
                self._ok_cells_n = ok_count
        if self._ok_cells_n > 0:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self._dot_prog['u_center'].value = (self.cx, self.cy)
            self._dot_prog['u_ppi'].value = self.ppi
            self._dot_prog['u_screen'].value = (float(self.w), float(self.h))
            self._dot_prog['u_square'].value = 1
            self._ok_cells_vao.render(moderngl.POINTS,
                                      vertices=self._ok_cells_n)
            self.ctx.disable(moderngl.PROGRAM_POINT_SIZE)
            self._dot_prog['u_square'].value = 0

    def _render_fetched(self):
        """Draw fetched image overlays when zoomed in."""
        if not (self._fetched_textures and self.ppi >= FETCH_MIN_PPI):
            return
        ppi = self.ppi
        for idx, (tex, gx, gy) in self._fetched_textures.items():
            sx0 = (gx - self.cx) * ppi + self.w / 2
            sy0 = (gy - self.cy) * ppi + self.h / 2
            sx1 = sx0 + ppi
            sy1 = sy0 + ppi
            if sx1 < 0 or sx0 > self.w or sy1 < 0 or sy0 > self.h:
                continue
            x0 = 2.0 * sx0 / self.w - 1.0
            x1 = 2.0 * sx1 / self.w - 1.0
            y0 = 1.0 - 2.0 * sy1 / self.h
            y1 = 1.0 - 2.0 * sy0 / self.h
            tex.use(0)
            self._overlay_prog['u_tex'].value = 0
            self._overlay_prog['u_rect'].value = (x0, y0, x1, y1)
            self._overlay_vao.render(moderngl.TRIANGLE_STRIP)

    def _render_dots(self):
        """Draw link-check status dots."""
        if not (self._show_dots and self._dot_n > 0 and self._dot_vao is not None):
            return
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._dot_prog['u_center'].value = (self.cx, self.cy)
        self._dot_prog['u_ppi'].value = self.ppi
        self._dot_prog['u_screen'].value = (float(self.w), float(self.h))
        self._dot_prog['u_square'].value = 0
        self._dot_vao.render(moderngl.POINTS, vertices=self._dot_n)
        self.ctx.disable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)

    def _render_preview(self):
        """Draw hi-res preview popup at full resolution near cursor."""
        if not self._show_preview or self.ppi < DETAIL_MIN_PPI:
            return
        idx = self._hover_idx
        if idx < 0 or idx not in self._fi_set:
            if self._preview_tex is not None:
                self._preview_tex.release()
                self._preview_tex = None
                self._preview_idx = -1
            return
        # build / update preview texture
        if idx != self._preview_idx:
            if self._preview_tex is not None:
                self._preview_tex.release()
                self._preview_tex = None
            pixels = self._load_fetched(idx)
            if pixels is None:
                self._preview_idx = -1
                return
            h, w = pixels.shape[:2]
            tex = self.ctx.texture((w, h), 3, pixels.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._preview_tex = tex
            self._preview_tw = w
            self._preview_th = h
            self._preview_idx = idx
        # position: bottom-right corner of popup anchored left-and-up from cursor
        mx, my = glfw.get_cursor_pos(self.win)
        tw, th = self._preview_tw, self._preview_th
        ox = mx - META_OFFSET[0] - tw
        oy = my - META_OFFSET[1] - th
        # clamp to screen
        ox = max(0, min(ox, self.w - tw))
        oy = max(0, min(oy, self.h - th))
        # convert screen coords (y-down) to NDC (y-up)
        x0 = 2.0 * ox / self.w - 1.0
        x1 = 2.0 * (ox + tw) / self.w - 1.0
        y0 = 1.0 - 2.0 * (oy + th) / self.h
        y1 = 1.0 - 2.0 * oy / self.h
        self._preview_tex.use(0)
        self._overlay_prog['u_tex'].value = 0
        self._overlay_prog['u_rect'].value = (x0, y0, x1, y1)
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)

    def _render_info_overlay(self):
        """Draw metadata or help overlay on top of everything."""
        overlay_tex = None
        overlay_tw = overlay_th = 0
        if self._show_help:
            self._render_help_texture()
            overlay_tex = self._help_tex
            overlay_tw, overlay_th = self._help_tw, self._help_th
        elif self._show_meta and self._hover_idx >= 0:
            self._render_meta_texture(self._hover_idx)
            overlay_tex = self._meta_tex
            overlay_tw, overlay_th = self._meta_tw, self._meta_th
        if overlay_tex is None:
            return
        mx, my = glfw.get_cursor_pos(self.win)
        # upper-left of overlay at cursor + offset (screen coords, y-down)
        ox = mx + META_OFFSET[0]
        oy = my + META_OFFSET[1]
        # clamp so overlay stays fully on-screen
        ox = min(ox, self.w - overlay_tw)
        oy = min(oy, self.h - overlay_th)
        ox = max(0, ox)
        oy = max(0, oy)
        # convert screen coords (y-down) to NDC (y-up)
        x0 = 2.0 * ox / self.w - 1.0
        x1 = 2.0 * (ox + overlay_tw) / self.w - 1.0
        y0 = 1.0 - 2.0 * (oy + overlay_th) / self.h
        y1 = 1.0 - 2.0 * oy / self.h
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        overlay_tex.use(0)
        self._overlay_prog['u_tex'].value = 0
        self._overlay_prog['u_rect'].value = (x0, y0, x1, y1)
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    # ── main loop ────────────────────────────────────────────────────

    def run(self):
        # Ctrl+C → graceful shutdown through the finally block
        def _sigint(sig, frame):
            glfw.set_window_should_close(self.win, True)
        signal.signal(signal.SIGINT, _sigint)
        try:
            while not glfw.window_should_close(self.win):
                if self._dirty:
                    self._dirty = False
                    self._render()
                    glfw.swap_buffers(self.win)

                # Pick up completed background build
                if self._upload_pending_detail():
                    self._dirty = True

                if self._link_dirty:
                    self._link_dirty = False
                    self._dot_needs_rebuild = True
                    self._dirty = True

                # Rebuild dot buffer only when needed (throttled)
                if self._dot_needs_rebuild:
                    now = time.time()
                    if now - self._dot_last_rebuild > 0.5:
                        old_n = self._dot_n
                        self._rebuild_dots()
                        self._dot_last_rebuild = now
                        self._dot_needs_rebuild = False
                        if self._dot_n != old_n or self._dot_journal_had_updates:
                            self._dirty = True

                # Upload completed texture loads (triggers render but not tex manager)
                if self._upload_pending_fetches():
                    self._dirty = True

                # Only check fetched textures on viewport change (not on uploads)
                if self._viewport_changed:
                    self._viewport_changed = False
                    self._manage_fetched_textures()

                # Kick off background build if needed (but don't restart one in flight)
                feasible = self._detail_feasible()
                if feasible and not self._detail_covers_viewport():
                    cur_vp = (self.cx, self.cy, self.ppi, self.w, self.h)
                    if (not (self._bg_thread and self._bg_thread.is_alive())
                            and self._det_build_vp != cur_vp):
                        self._start_detail_build()
                elif not feasible and self._det_tex is not None:
                    self._det_tex.release()
                    self._det_tex = None
                    self._det_key = None
                    self._dirty = True

                # Auto-stop crawl when both directions are done
                if self._crawl_threads and not any(t.is_alive() for t in self._crawl_threads):
                    self._stop_crawl()
                    self._dirty = True

                # Periodic status line during crawl
                if self._crawl_threads or self._status_shown:
                    now_s = time.monotonic()
                    if now_s - self._status_last >= 1.0:
                        self._status_last = now_s
                        self._print_status()

                # Poll faster while a build is in flight or crawl is active
                busy = ((self._bg_thread and self._bg_thread.is_alive())
                        or self._crawl_threads)
                glfw.wait_events_timeout(0.01 if busy else 0.05)
        finally:
            self._stop_crawl()
            self._link_status.flush()
            if self._shard_tf is not None:
                self._shard_tf.close()
            self._fetch_db.commit()
            self._fetch_db.close()
            self._link_pool.shutdown(wait=False, cancel_futures=True)
            self._release_lock()
            glfw.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactive viewer for the 80 Million Tiny Images dataset')
    parser.add_argument('--data-dir', default=None,
                        help=f'Path to data directory (default: {DATA_DIR})')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show per-image crawl/fetch output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress all output except fatal errors')
    parser.add_argument('--layout', choices=['hilbert', 'gilbert'],
                        default='gilbert',
                        help='Space-filling curve layout (default: gilbert)')
    args = parser.parse_args()
    Viewer(data_dir=args.data_dir, verbose=args.verbose,
           quiet=args.quiet, layout=args.layout).run()
