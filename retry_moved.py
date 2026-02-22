#!/usr/bin/env python3
"""One-time retry of all 'moved' URLs, following redirects to fetch images.

Run with the viewer stopped (shares the tar shards and sqlite DB).

Usage: pixi run python retry_moved.py [-n MAX] [-c CONNECTIONS] [-v]
"""
import argparse, io, os, signal, sqlite3, sys, tarfile, time, threading
import concurrent.futures
import socket
import urllib.request
import urllib.error
import numpy as np
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, 'data')

NUM_IMAGES = 79302017
META_REC = 768
FETCH_MAX_SIZE = 512
FETCH_SHARD_MAX = 1 << 31
LINK_TIMEOUT = 10
STATUS_CODES = {
    'pending': 1, 'ok': 2, 'moved': 3, 'not_found': 4,
    'error': 5, 'dns': 6, 'no_url': 7, 'not_image': 8,
}

# Counters
_lock = threading.Lock()
_stats = {'ok': 0, 'moved': 0, 'not_found': 0, 'error': 0,
          'dns': 0, 'not_image': 0, 'skipped': 0}
_done = 0
_total = 0
_stop = threading.Event()

# Shard state (protected by _shard_lock)
_shard_lock = threading.Lock()
_shard_num = 0
_shard_tf = None
_shard_dirty = 0

# Connection semaphore (set in main)
_net_sem = None


def _open_shard(fetch_dir):
    global _shard_tf, _shard_num
    if _shard_tf is not None:
        _shard_tf.close()
        _shard_tf = None
    shard = f"shard_{_shard_num:06d}.tar"
    path = os.path.join(fetch_dir, shard)
    try:
        _shard_tf = tarfile.open(path, 'a')
    except tarfile.ReadError:
        # Let the viewer handle repair; just start a new shard
        _shard_num += 1
        shard = f"shard_{_shard_num:06d}.tar"
        path = os.path.join(fetch_dir, shard)
        _shard_tf = tarfile.open(path, 'a')
    return shard


def save_fetched(idx, img, fetch_dir, db):
    global _shard_tf, _shard_num, _shard_dirty
    with _shard_lock:
        if db.execute('SELECT 1 FROM images WHERE idx=?', (idx,)).fetchone():
            return True
        jpeg_buf = io.BytesIO()
        img.save(jpeg_buf, format='JPEG', quality=90)
        jpeg_data = jpeg_buf.getvalue()
        w, h = img.size
        if _shard_tf is None:
            shard = _open_shard(fetch_dir)
        else:
            shard = f"shard_{_shard_num:06d}.tar"
        info = tarfile.TarInfo(name=f"{idx:010d}.jpg")
        info.size = len(jpeg_data)
        data_offset = _shard_tf.fileobj.tell() + 512
        _shard_tf.addfile(info, io.BytesIO(jpeg_data))
        pos = _shard_tf.fileobj.tell()
        if pos >= FETCH_SHARD_MAX:
            _shard_num += 1
            _open_shard(fetch_dir)
        db.execute(
            'INSERT INTO images (idx, shard, width, height, offset, size) '
            'VALUES (?,?,?,?,?,?)',
            (idx, shard, w, h, data_offset, len(jpeg_data)))
        _shard_dirty += 1
        if _shard_dirty >= 100:
            db.commit()
            _shard_dirty = 0
        return True


def save_redirect(idx, url, db):
    try:
        with _shard_lock:
            db.execute(
                'INSERT OR REPLACE INTO redirects (idx, url) VALUES (?, ?)',
                (idx, url))
    except Exception:
        pass


def retry_one(idx, url, link_status, fetch_dir, db, verbose):
    global _done
    if _stop.is_set():
        return
    try:
        headers = {'User-Agent': 'TinyImagesViewer/1.0'}
        with _net_sem:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=LINK_TIMEOUT)
            code = resp.getcode()
            if not (200 <= code < 300):
                link_status[idx] = STATUS_CODES['error']
                with _lock:
                    _stats['error'] += 1
                return
            # Save final URL if it differs from original (redirect chain resolved)
            final_url = resp.geturl()
            if final_url != url:
                save_redirect(idx, final_url, db)
            ctype = resp.headers.get('Content-Type', '')
            if not ctype.startswith('image/'):
                # Still moved, just not an image at destination
                with _lock:
                    _stats['not_image'] += 1
                return
            raw = resp.read(16 * 1024 * 1024)
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        if s > FETCH_MAX_SIZE:
            img = img.resize((FETCH_MAX_SIZE, FETCH_MAX_SIZE), Image.LANCZOS)
        if save_fetched(idx, img, fetch_dir, db):
            link_status[idx] = STATUS_CODES['ok']
            with _lock:
                _stats['ok'] += 1
            if verbose:
                print(f"  #{idx:,} ok {w}x{h}", flush=True)
        else:
            with _lock:
                _stats['error'] += 1
    except urllib.error.HTTPError as e:
        if e.code == 404:
            link_status[idx] = STATUS_CODES['not_found']
            with _lock:
                _stats['not_found'] += 1
        else:
            with _lock:
                _stats['error'] += 1
    except socket.gaierror:
        link_status[idx] = STATUS_CODES['dns']
        with _lock:
            _stats['dns'] += 1
    except Exception:
        with _lock:
            _stats['error'] += 1
    finally:
        with _lock:
            _done += 1


def main():
    global _shard_num, _total, _net_sem

    parser = argparse.ArgumentParser(description='Retry moved URLs with redirect following')
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('-n', '--max', type=int, default=0,
                        help='Max URLs to retry (0=all)')
    parser.add_argument('-c', '--connections', type=int, default=90,
                        help='Max concurrent HTTP connections (default: 90)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    _net_sem = threading.Semaphore(args.connections)

    dd = args.data_dir
    fetch_dir = os.path.join(dd, 'fetched')

    # Check lock
    lock_path = os.path.join(fetch_dir, 'viewer.pid')
    if os.path.exists(lock_path):
        try:
            pid = int(open(lock_path).read().strip())
            os.kill(pid, 0)
            print(f"Viewer is running (PID {pid}). Stop it first.", file=sys.stderr)
            sys.exit(1)
        except (ValueError, ProcessLookupError, OSError):
            pass

    meta = np.memmap(os.path.join(dd, 'tiny_metadata.bin'), dtype='uint8',
                     mode='r', shape=(NUM_IMAGES, META_REC))
    link_status = np.memmap(os.path.join(dd, 'link_status.bin'), dtype='uint8',
                            mode='r+', shape=(NUM_IMAGES,))

    moved_indices = np.where(link_status == STATUS_CODES['moved'])[0]
    print(f"Found {len(moved_indices):,} moved images")

    if args.max > 0:
        moved_indices = moved_indices[:args.max]
        print(f"Retrying first {len(moved_indices):,}")

    # Build URL list, skip those without URLs
    work = []
    for idx in moved_indices:
        url = bytes(meta[idx, 422:720]).decode('ascii', errors='replace').strip().rstrip('\x00')
        if not url:
            continue
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        work.append((int(idx), url))
    _total = len(work)
    print(f"{_total:,} URLs to retry ({args.connections} concurrent connections)")

    # Open DB and find current shard
    db = sqlite3.connect(os.path.join(fetch_dir, 'index.db'), check_same_thread=False)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS redirects (
        idx INTEGER PRIMARY KEY,
        url TEXT NOT NULL)''')
    db.commit()
    shard_files = sorted(
        f for f in os.listdir(fetch_dir)
        if f.startswith('shard_') and f.endswith('.tar'))
    if shard_files:
        _shard_num = int(shard_files[-1].split('_')[1].split('.')[0])
        if os.path.getsize(os.path.join(fetch_dir, shard_files[-1])) >= FETCH_SHARD_MAX:
            _shard_num += 1

    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    t0 = time.time()
    # 4× connections for worker threads — plenty of headroom for I/O + processing
    n_workers = min(args.connections * 4, len(work))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for idx, url in work:
            if _stop.is_set():
                break
            futures.append(pool.submit(retry_one, idx, url, link_status,
                                       fetch_dir, db, args.verbose))

        # Progress reporting
        while not _stop.is_set():
            not_done = [f for f in futures if not f.done()]
            if not not_done:
                break
            with _lock:
                d, t = _done, _total
                s = dict(_stats)
            elapsed = time.time() - t0
            rate = d / elapsed if elapsed > 0 else 0
            eta = (t - d) / rate if rate > 0 else 0
            print(f"\r  {d:,}/{t:,} ({100*d/t:.1f}%)  "
                  f"ok:{s['ok']} !img:{s['not_image']} 404:{s['not_found']} "
                  f"err:{s['error']} dns:{s['dns']}  "
                  f"{rate:.0f}/s ETA {eta:.0f}s\033[K",
                  end='', flush=True)
            time.sleep(1.0)

    # Final commit
    with _shard_lock:
        if _shard_tf is not None:
            _shard_tf.close()
        db.commit()
        db.close()
    link_status.flush()

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.0f}s")
    for k, v in sorted(_stats.items()):
        if v:
            print(f"  {k}: {v:,}")


if __name__ == '__main__':
    main()
