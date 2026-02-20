# tiny_metadata.bin — Record Format

57 GB file containing **79,302,017** fixed-width ASCII records, one per image,
matching the order of `tiny_images.bin`.

Each record is **768 bytes**, space-padded, no newlines.

## Field Layout

| Offset | Width | Field | Format | Description |
|-------:|------:|-------|--------|-------------|
| 0 | 80 | `keyword` | space-padded | Search query term (lowercase) |
| 80 | 80 | `filename` | space-padded | e.g. `a-bomb_s_000001.png` |
| 160 | 15 | | | Padding (always spaces) |
| 175 | 5 | `image_spec` | `WHC` | Image dimensions, usually `32323` (32×32×3); `32321` for grayscale (~2.7%) |
| 180 | 32 | `crawl_date` | left-justified | e.g. `15 May 2006 12:39:30 +0000` |
| 212 | 10 | `search_engine` | space-padded | One of 7 engines (see below) |
| 222 | 200 | `thumbnail_url` | space-padded | Search-engine thumbnail or cache URL |
| 422 | 298 | `source_url` | space-padded | Original image URL (long URLs overflow past byte 560) |
| 720 | 30 | | | Padding (always spaces) |
| 750 | 4 | `page_number` | `%-4d` | Search-results page number |
| 754 | 4 | `page_position` | `%-4d` | Position on that page (resets per engine) |
| 758 | 4 | `engine_cum_pos` | `%-4d` | Cumulative position within engine for this keyword |
| 762 | 4 | `filename_index` | `%-4d` | Image number within keyword (matches `_s_NNNNNN` in filename) |
| 766 | 2 | `sentinel` | `%-2d` | Always `-1` |

The five trailing numeric fields are left-justified integers written with
C-style `printf("%-4d%-4d%-4d%-4d%-2d", ...)`.  When `filename_index` reaches
four digits it abuts the sentinel with no separator (e.g. `1091-1`); use
`split()` or the fixed offsets above to parse.

## Search Engines

Seven image search engines were crawled (May–November 2006):

| Engine | Notes |
|--------|-------|
| `google` | Google Images |
| `altavista` | AltaVista (via Yahoo image search) |
| `ask` | Ask.com |
| `flickr` | Flickr |
| `cydral` | Cydral image search |
| `picsearch` | Picsearch |
| `webshots` | Webshots |

## Crawl Order

Within each keyword the records cycle through engines round-robin by page:

1. google page 1 (~20 results)
2. altavista page 1
3. ask page 1
4. flickr page 1
5. cydral page 1
6. picsearch page 1
7. webshots page 1
8. google page 2
9. altavista page 2
10. …and so on

`filename_index` increments monotonically across all engines for the keyword,
while `engine_cum_pos` resets to 1 each time the engine changes.

## Quick Access (Python)

```python
import numpy as np

META = 'data/tiny_metadata.bin'
REC  = 768

meta = np.memmap(META, dtype='uint8', mode='r')

def read_record(i):
    r = bytes(meta[i * REC : (i + 1) * REC])
    return {
        'keyword':        r[  0: 80].decode('ascii').rstrip(),
        'filename':       r[ 80:160].decode('ascii').rstrip(),
        'image_spec':     r[175:180].decode('ascii'),
        'crawl_date':     r[180:212].decode('ascii').strip(),
        'search_engine':  r[212:222].decode('ascii').strip(),
        'thumbnail_url':  r[222:422].decode('ascii').strip(),
        'source_url':     r[422:720].decode('ascii').strip(),
        'page_number':    int(r[750:754]),
        'page_position':  int(r[754:758]),
        'engine_cum_pos': int(r[758:762]),
        'filename_index': int(r[762:766]),
    }
```
