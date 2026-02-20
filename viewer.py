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
  L          toggle link-check dots and fetched overlays
  C          toggle Hilbert crawl from hovered image
  D          dim all except fetched images (find live sources)
"""

import math, os, signal, sys, time, threading
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
FILENAME = '/data-pool/tor/tiny/data/tiny_images.bin'
METAFILE = '/data-pool/tor/tiny/data/tiny_metadata.bin'
NUM_IMAGES = 79302017
IMG = 32
CH = 3
BPI = IMG * IMG * CH
META_REC = 768          # bytes per metadata record
META_KW_LEN = 80        # keyword field width
ORDER = 14
SIDE = 1 << ORDER          # 16384
AVG_CACHE  = os.path.join(os.path.dirname(FILENAME), 'avg_colors.npy')
GRID_CACHE = os.path.join(os.path.dirname(FILENAME), 'hilbert_avg_grid.npy')


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
STATUS_CODES = {
    'pending': 1, 'ok': 2, 'moved': 3, 'not_found': 4,
    'error': 5, 'dns': 6, 'no_url': 7, 'not_image': 8,
}
STATUS_NAMES = {v: k for k, v in STATUS_CODES.items()}
LINK_CACHE = os.path.join(os.path.dirname(FILENAME), 'link_status.bin')
FETCH_MAX_SIZE = 512
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
out vec3 v_color;
void main() {
    vec2 center = grid_pos + 0.5;
    vec2 sp = (center - u_center) * u_ppi + u_screen * 0.5;
    sp.y = u_screen.y - sp.y;
    vec2 ndc = sp / u_screen * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = clamp(u_ppi * 0.4, 6.0, 32.0);
    v_color = color;
}
"""

DOT_FRAG = """
#version 330
in vec3 v_color;
out vec4 frag;
void main() {
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    frag = vec4(v_color, 1.0);
}
"""


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


def load_or_build_grid(data):
    """Load the SIDE×SIDE×3 Hilbert avg-colour grid, building it if needed."""
    if os.path.exists(GRID_CACHE):
        print("Loading cached Hilbert avg grid …")
        return np.load(GRID_CACHE)

    # fast path: build from per-image avg cache (no dataset scan)
    if os.path.exists(AVG_CACHE):
        print("Building Hilbert avg grid from avg cache …")
        avg = np.load(AVG_CACHE)
        grid = _scatter_avg_to_grid(avg)
        np.save(GRID_CACHE, grid)
        print(f"Saved to {GRID_CACHE}")
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
    np.save(GRID_CACHE, grid)
    print(f"Saved to {GRID_CACHE}")
    return grid


# ── Viewer ────────────────────────────────────────────────────────────
class Viewer:
    @staticmethod
    def _fit_zoom(w, h):
        """Zoom level that fits the whole grid in 90% of the shorter axis."""
        ppi = 0.9 * min(w / SIDE, h / SIDE)
        return math.log2(max(ppi, 2 ** (MIN_ZOOM + 5))) - 5

    def __init__(self):
        # view state
        self.w, self.h = 1920, 1080
        self.cx = SIDE / 2.0
        self.cy = SIDE / 2.0
        self.zoom = self._fit_zoom(self.w, self.h)
        self.dragging = False
        self.mx = self.my = 0.0

        # detail texture cache
        self._det_key = None
        self._det_tex = None
        self._dirty = True        # need re-render

        # background detail builder
        self._bg_thread = None
        self._bg_cancel = threading.Event()
        self._bg_lock = threading.Lock()
        self._bg_results = []    # [(key, buf, vw, vh), ...] from bg thread

        # keyword search (None = inactive, str = active query)
        self._search = None

        # fullscreen state
        self._fullscreen = False
        self._windowed_pos = (100, 100)
        self._windowed_size = (self.w, self.h)

        # ── GLFW ──
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

        # ── moderngl ──
        self.ctx = moderngl.create_context()

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.ctx.buffer(quad), '2f', 'pos')]
        )

        # ── data ──
        print(f"Memory-mapping {FILENAME} …")
        self.data = np.memmap(FILENAME, dtype='uint8', mode='r',
                              shape=(NUM_IMAGES, CH, IMG, IMG))

        avg_grid = load_or_build_grid(self.data)
        self.avg_tex = self.ctx.texture((SIDE, SIDE), 3, avg_grid.tobytes())
        del avg_grid
        self.avg_tex.build_mipmaps()
        self.avg_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        # 1×1 dummy for the detail slot when unused
        self._dummy = self.ctx.texture((1, 1), 3, b'\x00\x00\x00')

        # ── metadata ──
        self.meta = np.memmap(METAFILE, dtype='uint8', mode='r',
                              shape=(NUM_IMAGES, META_REC))
        self._hover_idx = -1      # image index under cursor

        # ── metadata overlay ──
        self._show_meta = False
        self._meta_tex = None
        self._meta_idx = -1       # avoid redundant re-renders for same image
        self._meta_tw = 0         # texture width  (pixels)
        self._meta_th = 0         # texture height (pixels)

        self._overlay_prog = self.ctx.program(
            vertex_shader=OVERLAY_VERT, fragment_shader=OVERLAY_FRAG)
        overlay_quad = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype='f4')
        self._overlay_vao = self.ctx.vertex_array(
            self._overlay_prog,
            [(self.ctx.buffer(overlay_quad), '2f', 'pos')])

        # ── link checking ──
        self._link_lock = threading.Lock()
        self._link_pool = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        self._link_dirty = False
        self._click_sx = 0.0
        self._click_sy = 0.0
        self._show_dots = True

        # persistent link status (mmap'd byte array)
        if not os.path.exists(LINK_CACHE):
            fp = np.memmap(LINK_CACHE, dtype='uint8', mode='w+',
                           shape=(NUM_IMAGES,))
            del fp
        self._link_status = np.memmap(LINK_CACHE, dtype='uint8', mode='r+',
                                      shape=(NUM_IMAGES,))
        self._link_checks = {}
        snapshot = np.array(self._link_status)   # copy — nonzero on mmap can race
        checked = np.nonzero(snapshot)[0]
        del snapshot
        for i in checked:
            self._link_checks[int(i)] = STATUS_NAMES.get(
                int(self._link_status[i]), 'error')
        if checked.size:
            print(f"Restored {checked.size:,} link checks from cache",
                  flush=True)

        # Hilbert crawl state
        self._crawl_cancel = threading.Event()
        self._crawl_threads = []
        self._crawl_origin = -1
        self._crawl_count = [0, 0]    # [forward, backward]

        # dim mode — darken everything except fetched overlays
        self._dim_mode = False

        # fetched image overlay
        self._fetched_queue = []      # [(idx, gx, gy, pixels), ...] from workers
        self._fetched_lock = threading.Lock()
        self._fetched_textures = {}   # idx → (texture, gx, gy)

        # dot shader
        self._dot_prog = self.ctx.program(
            vertex_shader=DOT_VERT, fragment_shader=DOT_FRAG)

        print(f"Hilbert grid {SIDE}×{SIDE} (order {ORDER}) = {NUM_IMAGES:,} images", flush=True)
        print("Scroll=zoom  Drag=pan  Click=check  C=crawl  D=dim  F=full  M=meta  L=dots  /=search  Esc/Q=quit", flush=True)

    # ── properties ────────────────────────────────────────────────────
    @property
    def ppi(self):
        """Display pixels per image at current zoom."""
        return 2.0 ** (self.zoom + 5)

    # ── helpers ────────────────────────────────────────────────────────
    def _screen_to_image_idx(self, sx, sy):
        """Return the linear image index under screen coords (sx, sy), or -1."""
        ppi = self.ppi
        gx = self.cx + (sx - self.w / 2) / ppi
        gy = self.cy + (sy - self.h / 2) / ppi
        ix, iy = int(math.floor(gx)), int(math.floor(gy))
        if ix < 0 or iy < 0 or ix >= SIDE or iy >= SIDE:
            return -1
        idx = int(hilbert_xy2d(ORDER, ix, iy))
        return idx if idx < NUM_IMAGES else -1

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

        buf = img.tobytes()
        if self._meta_tex is not None:
            self._meta_tex.release()
        self._meta_tex = self.ctx.texture((tw, th), 4, buf)
        self._meta_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._meta_tw = tw
        self._meta_th = th
        self._meta_idx = idx

    # ── keyword search ────────────────────────────────────────────────
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
        gx, gy = hilbert_d2xy(ORDER, np.int64(idx))
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
    def _enqueue_link_check(self, idx):
        """Queue a HEAD request for the source URL of image *idx*."""
        with self._link_lock:
            if idx in self._link_checks:
                return
            self._link_checks[idx] = 'pending'
        self._dirty = True
        self._link_pool.submit(self._check_link, idx)

    def _check_link(self, idx):
        """HEAD-request the source URL and classify the result (runs in thread pool)."""
        meta = self._read_meta(idx)
        url = meta['source_url'].strip('\x00 ')
        if not url:
            status = 'no_url'
        else:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            try:
                req = urllib.request.Request(url, method='HEAD')
                opener = urllib.request.build_opener(_NoRedirectHandler)
                resp = opener.open(req, timeout=LINK_TIMEOUT)
                code = resp.getcode()
                status = 'ok' if 200 <= code < 300 else 'error'
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 307, 308):
                    status = 'moved'
                elif e.code == 404:
                    status = 'not_found'
                else:
                    status = 'error'
            except socket.gaierror:
                status = 'dns'
            except Exception:
                status = 'error'
        with self._link_lock:
            self._link_checks[idx] = status
            self._link_status[idx] = STATUS_CODES.get(status, 5)
        self._link_dirty = True
        if status == 'ok':
            self._fetch_image(idx, url)

    def _fetch_image(self, idx, url):
        """GET the image at *url*, center-crop square, and queue for GPU upload."""
        try:
            print(f"  fetch #{idx} GET {url[:80]}", flush=True)
            req = urllib.request.Request(
                url, headers={'User-Agent': 'TinyImagesViewer/1.0'})
            resp = urllib.request.urlopen(req, timeout=LINK_TIMEOUT)
            ctype = resp.headers.get('Content-Type', '')
            if not ctype.startswith('image/'):
                print(f"  fetch #{idx} not image: {ctype}", flush=True)
                with self._link_lock:
                    self._link_checks[idx] = 'not_image'
                    self._link_status[idx] = STATUS_CODES['not_image']
                self._link_dirty = True
                return
            raw = resp.read(16 * 1024 * 1024)  # 16 MB cap
            print(f"  fetch #{idx} got {len(raw)} bytes", flush=True)
            img = Image.open(io.BytesIO(raw)).convert('RGB')
            w, h = img.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
            if s > FETCH_MAX_SIZE:
                img = img.resize((FETCH_MAX_SIZE, FETCH_MAX_SIZE), Image.LANCZOS)
            pixels = np.asarray(img, dtype='uint8').copy()
            print(f"  fetch #{idx} image {w}x{h} → {pixels.shape}", flush=True)
            gx, gy = hilbert_d2xy(ORDER, np.int64(idx))
            with self._fetched_lock:
                self._fetched_queue.append((idx, int(gx), int(gy), pixels))
            self._link_dirty = True
        except Exception as e:
            print(f"  fetch #{idx} FAILED: {e}", flush=True)

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
                    print(f"  crawl {label} stopped: {CRAWL_STOP_RUN} consecutive already-checked at #{idx:,}", flush=True)
                    break
                continue   # skip already-checked instantly
            consec = 0
            self._enqueue_link_check(idx)
            self._crawl_count[slot] += 1
            self._crawl_cancel.wait(0.05)

    def _start_crawl(self, idx):
        self._stop_crawl()
        self._crawl_cancel = threading.Event()
        self._crawl_origin = idx
        self._crawl_count = [0, 0]
        fwd = threading.Thread(target=self._crawl_worker, args=(idx, +1), daemon=True)
        bwd = threading.Thread(target=self._crawl_worker, args=(idx, -1), daemon=True)
        self._crawl_threads = [fwd, bwd]
        fwd.start()
        bwd.start()
        print(f"Crawl started from #{idx:,}", flush=True)

    def _stop_crawl(self):
        if self._crawl_threads:
            self._crawl_cancel.set()
            for t in self._crawl_threads:
                t.join(timeout=1.0)
            total = sum(self._crawl_count)
            if total:
                print(f"Crawl stopped ({total:,} checked)", flush=True)
            self._crawl_threads = []

    # ── input callbacks ───────────────────────────────────────────────
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
                        self._enqueue_link_check(idx)

    def _on_cursor(self, win, x, y):
        if self.dragging:
            ppi = self.ppi
            self.cx -= (x - self.mx) / ppi
            self.cy -= (y - self.my) / ppi
            self.mx, self.my = x, y
            self._dirty = True
        self._update_title(x, y)
        if self._show_meta:
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
            self.cx, self.cy = SIDE / 2.0, SIDE / 2.0
            self.zoom = self._fit_zoom(self.w, self.h)
            self._dirty = True
        elif key == glfw.KEY_F:
            self._toggle_fullscreen()
        elif key == glfw.KEY_M:
            self._show_meta = not self._show_meta
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

    # ── detail texture ────────────────────────────────────────────────
    def _viewport_rect(self):
        """Return (gx0, gy0, gx1, gy1) of the current viewport in grid coords."""
        ppi = self.ppi
        hw = self.w / (2 * ppi)
        hh = self.h / (2 * ppi)
        return (self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)

    def _detail_feasible(self):
        """Can we have a detail texture at the current zoom?"""
        return self.ppi >= DETAIL_MIN_PPI

    def _detail_covers_viewport(self):
        """Does the cached detail texture fully cover the current viewport?"""
        if not self._det_key or self._det_tex is None:
            return False
        vp = self._viewport_rect()
        cx0, cy0, cx1, cy1 = self._det_key
        return cx0 <= vp[0] and cy0 <= vp[1] and cx1 >= vp[2] and cy1 >= vp[3]

    @staticmethod
    def _build_region(data, gx0, gy0, gx1, gy1, cancel=None):
        """Assemble a pixel buffer for a grid region (Hilbert layout)."""
        vw, vh = gx1 - gx0, gy1 - gy0
        buf = np.full((vh * IMG, vw * IMG, CH), 13, dtype='uint8')

        # grid coords for every cell in the region
        gxs, gys = np.meshgrid(
            np.arange(gx0, gx1, dtype=np.int64),
            np.arange(gy0, gy1, dtype=np.int64))
        gxs = gxs.ravel()
        gys = gys.ravel()

        # Hilbert index for each cell
        file_idx = hilbert_xy2d(ORDER, gxs, gys)

        # filter valid
        valid = file_idx < NUM_IMAGES
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

        # read each contiguous run
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
            # scatter into output buffer
            lx = gxs[rs:re] - gx0
            ly = gys[rs:re] - gy0
            for k in range(re - rs):
                buf[int(ly[k]) * IMG:(int(ly[k]) + 1) * IMG,
                    int(lx[k]) * IMG:(int(lx[k]) + 1) * IMG] = chunk[k]

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

        def clamp_region(gx0, gy0, gx1, gy1):
            max_imgs = MAX_TEX // IMG
            gx0 = max(0, gx0); gy0 = max(0, gy0)
            gx1 = min(SIDE, gx1); gy1 = min(SIDE, gy1)
            vw, vh = gx1 - gx0, gy1 - gy0
            if vw > max_imgs:
                ex = vw - max_imgs
                gx0 += ex // 2; gx1 = gx0 + max_imgs
            if vh > max_imgs:
                ex = vh - max_imgs
                gy0 += ex // 2; gy1 = gy0 + max_imgs
            return gx0, gy0, gx1, gy1

        def worker():
            hw = w / (2 * ppi)
            hh = h / (2 * ppi)

            # Phase 1: viewport only (small, fast)
            vgx0, vgy0, vgx1, vgy1 = clamp_region(
                int(cx - hw) - 1, int(cy - hh) - 1,
                int(cx + hw) + 2, int(cy + hh) + 2)
            if vgx1 > vgx0 and vgy1 > vgy0:
                buf, vw, vh = Viewer._build_region(
                    data, vgx0, vgy0, vgx1, vgy1, cancel=cancel)
                if cancel.is_set():
                    return
                with self._bg_lock:
                    self._bg_results.append(
                        ((vgx0, vgy0, vgx1, vgy1), buf, vw, vh))

            if cancel.is_set():
                return

            # Phase 2: full margin region
            mw = hw * DETAIL_MARGIN
            mh = hh * DETAIL_MARGIN
            mgx0, mgy0, mgx1, mgy1 = clamp_region(
                int(cx - mw), int(cy - mh),
                int(cx + mw) + 1, int(cy + mh) + 1)
            if mgx1 > mgx0 and mgy1 > mgy0:
                buf, vw, vh = Viewer._build_region(
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

    def _upload_pending_fetches(self):
        """Upload queued fetched images as GPU textures (main thread only)."""
        with self._fetched_lock:
            queue = list(self._fetched_queue)
            self._fetched_queue.clear()
        if not queue:
            return False
        for idx, gx, gy, pixels in queue:
            h, w = pixels.shape[:2]
            tex = self.ctx.texture((w, h), 3, pixels.tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            if idx in self._fetched_textures:
                self._fetched_textures[idx][0].release()
            self._fetched_textures[idx] = (tex, gx, gy)
            print(f"  uploaded #{idx} tex {w}x{h} at grid ({gx},{gy})", flush=True)
        return True

    # ── render ────────────────────────────────────────────────────────
    def _render(self):
        """Render immediately with whatever detail texture is cached."""
        self.ctx.clear(0.05, 0.05, 0.05)

        # use cached detail (may be stale / partially offscreen — that's fine,
        # the shader falls through to avg colours for uncovered areas)
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
        self.prog['u_grid'].value   = (float(SIDE), float(SIDE))
        self.prog['u_dim'].value    = 0.85 if self._dim_mode else 0.0

        self.vao.render(moderngl.TRIANGLE_STRIP)

        # ── fetched image overlays ──
        if self._fetched_textures and self.ppi >= FETCH_MIN_PPI:
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

        # ── metadata overlay ──
        if self._show_meta and self._hover_idx >= 0:
            self._render_meta_texture(self._hover_idx)
            if self._meta_tex is not None:
                mx, my = glfw.get_cursor_pos(self.win)
                # upper-left of overlay at cursor + offset (screen coords, y-down)
                ox = mx + META_OFFSET[0]
                oy = my + META_OFFSET[1]
                # clamp so overlay stays fully on-screen
                ox = min(ox, self.w - self._meta_tw)
                oy = min(oy, self.h - self._meta_th)
                ox = max(0, ox)
                oy = max(0, oy)
                # convert screen coords (y-down) to NDC (y-up)
                x0 = 2.0 * ox / self.w - 1.0
                x1 = 2.0 * (ox + self._meta_tw) / self.w - 1.0
                y0 = 1.0 - 2.0 * (oy + self._meta_th) / self.h
                y1 = 1.0 - 2.0 * oy / self.h
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = (
                    moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                self._meta_tex.use(0)
                self._overlay_prog['u_tex'].value = 0
                self._overlay_prog['u_rect'].value = (x0, y0, x1, y1)
                self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
                self.ctx.disable(moderngl.BLEND)

        # ── link-check dots ──
        if self._show_dots and self._link_checks:
            with self._link_lock:
                checks = dict(self._link_checks)
            n = len(checks)
            indices = np.array(list(checks.keys()), dtype=np.int64)
            gx, gy = hilbert_d2xy(ORDER, indices)
            vdata = np.empty((n, 5), dtype='f4')
            vdata[:, 0] = gx.astype('f4')
            vdata[:, 1] = gy.astype('f4')
            for i, idx in enumerate(checks):
                vdata[i, 2:5] = DOT_COLORS.get(checks[idx], (0.5, 0.5, 0.5))
            buf = self.ctx.buffer(vdata.tobytes())
            vao = self.ctx.vertex_array(
                self._dot_prog, [(buf, '2f 3f', 'grid_pos', 'color')])
            self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._dot_prog['u_center'].value = (self.cx, self.cy)
            self._dot_prog['u_ppi'].value = self.ppi
            self._dot_prog['u_screen'].value = (float(self.w), float(self.h))
            vao.render(moderngl.POINTS)
            self.ctx.disable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
            vao.release()
            buf.release()

    # ── main loop ─────────────────────────────────────────────────────
    def run(self):
        # Ctrl+C → graceful shutdown through the finally block
        def _sigint(sig, frame):
            glfw.set_window_should_close(self.win, True)
        signal.signal(signal.SIGINT, _sigint)
        try:
            while not glfw.window_should_close(self.win):
                if self._dirty:
                    self._dirty = False
                    self._render()              # instant — uses stale cache
                    glfw.swap_buffers(self.win)

                # Pick up completed background build
                if self._upload_pending_detail():
                    self._dirty = True          # re-render with fresh detail

                if self._link_dirty:
                    self._link_dirty = False
                    self._dirty = True

                if self._upload_pending_fetches():
                    self._dirty = True

                # Kick off background build if needed
                feasible = self._detail_feasible()
                if feasible and not self._detail_covers_viewport():
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

                # Poll faster while a build is in flight or crawl is active
                busy = ((self._bg_thread and self._bg_thread.is_alive())
                        or self._crawl_threads)
                glfw.wait_events_timeout(0.01 if busy else 0.05)
        finally:
            self._stop_crawl()
            self._link_status.flush()
            self._link_pool.shutdown(wait=False, cancel_futures=True)
            glfw.terminate()


if __name__ == '__main__':
    Viewer().run()
