#!/usr/bin/env python3
"""Interactive viewer for the 80 Million Tiny Images dataset.

Controls:
  Scroll     zoom (toward cursor)
  Drag       pan
  F          toggle fullscreen
  Escape     windowed (if fullscreen) / quit
  Q          quit
  Home       reset view
  /          search keywords (type to search, Esc/Enter to cancel)
"""

import math, os, sys, time, threading
import numpy as np
import glfw
import moderngl

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
AVG_CACHE = os.path.join(os.path.dirname(FILENAME), 'avg_colors.npy')


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
            return;
        }
    }

    // ── average-colour path ──
    vec2 uv = (vec2(cell) + 0.5) / u_grid;
    frag = texture(u_avg, uv);
}
"""


# ── Average-colour cache ─────────────────────────────────────────────
def load_or_compute_averages(data):
    if os.path.exists(AVG_CACHE):
        print("Loading cached average colours …")
        return np.load(AVG_CACHE)

    print("Computing average colours (one-time) …")
    avg = np.empty((NUM_IMAGES, 3), dtype='uint8')
    chunk = 200_000
    t0 = time.time()
    for i in range(0, NUM_IMAGES, chunk):
        e = min(i + chunk, NUM_IMAGES)
        avg[i:e] = data[i:e].mean(axis=(2, 3)).astype('uint8')
        elapsed = time.time() - t0
        pct = 100 * e / NUM_IMAGES
        eta = elapsed / pct * (100 - pct) if pct > 0 else 0
        print(f"\r  {pct:5.1f}%  ETA {eta:.0f}s", end='', flush=True)
    print()
    np.save(AVG_CACHE, avg)
    print(f"Saved to {AVG_CACHE}")
    return avg


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

        avg = load_or_compute_averages(self.data)

        # pack into a SIDE × SIDE RGB texture (Hilbert scatter)
        print("Building Hilbert avg texture …")
        avg_grid = np.full((SIDE, SIDE, 3), 13, dtype='uint8')
        CHUNK = 4_000_000
        for i in range(0, NUM_IMAGES, CHUNK):
            e = min(i + CHUNK, NUM_IMAGES)
            hx, hy = hilbert_d2xy(ORDER, np.arange(i, e, dtype=np.int64))
            avg_grid[hy, hx] = avg[i:e]
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

        print(f"Hilbert grid {SIDE}×{SIDE} (order {ORDER}) = {NUM_IMAGES:,} images")
        print("Scroll=zoom  Drag=pan  F=fullscreen  Esc=windowed/quit  /=search  Q=quit")

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
        if self._search is not None:
            parts = [f"/{self._search}"]
            if hover:
                parts.append(hover)
            glfw.set_window_title(self.win, "  |  ".join(parts))
        elif hover:
            glfw.set_window_title(self.win, hover)
        else:
            glfw.set_window_title(self.win, "Tiny Images")

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
            self.dragging = (action == glfw.PRESS)
            if self.dragging:
                self.mx, self.my = glfw.get_cursor_pos(win)

    def _on_cursor(self, win, x, y):
        if self.dragging:
            ppi = self.ppi
            self.cx -= (x - self.mx) / ppi
            self.cy -= (y - self.my) / ppi
            self.mx, self.my = x, y
            self._dirty = True
        self._update_title(x, y)

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

        self.vao.render(moderngl.TRIANGLE_STRIP)

    # ── main loop ─────────────────────────────────────────────────────
    def run(self):
        try:
            while not glfw.window_should_close(self.win):
                if self._dirty:
                    self._dirty = False
                    self._render()              # instant — uses stale cache
                    glfw.swap_buffers(self.win)

                # Pick up completed background build
                if self._upload_pending_detail():
                    self._dirty = True          # re-render with fresh detail

                # Kick off background build if needed
                feasible = self._detail_feasible()
                if feasible and not self._detail_covers_viewport():
                    self._start_detail_build()
                elif not feasible and self._det_tex is not None:
                    self._det_tex.release()
                    self._det_tex = None
                    self._det_key = None
                    self._dirty = True

                # Poll faster while a build is in flight
                building = self._bg_thread and self._bg_thread.is_alive()
                glfw.wait_events_timeout(0.01 if building else 0.05)
        finally:
            glfw.terminate()


if __name__ == '__main__':
    Viewer().run()
