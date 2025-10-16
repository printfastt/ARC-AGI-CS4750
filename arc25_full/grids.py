
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Iterable
from collections import Counter, deque

Grid = List[List[int]]

def shape(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)

def clone(g: Grid) -> Grid:
    return [row[:] for row in g]

def zeros(h: int, w: int, val: int = 0) -> Grid:
    return [[val for _ in range(w)] for _ in range(h)]

def full_like(g: Grid, val: int = 0) -> Grid:
    h,w = shape(g)
    return zeros(h,w,val)

def within(g: Grid, r: int, c: int) -> bool:
    return 0 <= r < len(g) and 0 <= c < len(g[0])

def eq(g1: Grid, g2: Grid) -> bool:
    if shape(g1) != shape(g2): return False
    for r in range(len(g1)):
        if g1[r] != g2[r]: return False
    return True

def transpose(g: Grid) -> Grid:
    h,w = shape(g)
    return [[g[r][c] for r in range(h)] for c in range(w)]

def rot90(g: Grid, k: int = 1) -> Grid:
    k %= 4
    out = g
    for _ in range(k):
        out = [list(row) for row in zip(*out)][::-1]
    return out

def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]

def flip_v(g: Grid) -> Grid:
    return g[::-1]

def histogram(g: Grid) -> Counter:
    return Counter(v for row in g for v in row)

def argmax_counter(c: Counter) -> int:
    return c.most_common(1)[0][0] if c else 0

def bbox_of(g: Grid, nonzero_only: bool = True) -> Optional[Tuple[int,int,int,int]]:
    rmin, rmax, cmin, cmax = 10**9, -1, 10**9, -1
    for r,row in enumerate(g):
        for c,v in enumerate(row):
            if (not nonzero_only) or (v != 0):
                rmin = min(rmin, r); rmax = max(rmax, r)
                cmin = min(cmin, c); cmax = max(cmax, c)
    if rmax == -1:
        return None
    return (rmin, rmax, cmin, cmax)

def crop(g: Grid, box: Tuple[int,int,int,int]) -> Grid:
    rmin,rmax,cmin,cmax = box
    return [row[cmin:cmax+1] for row in g[rmin:rmax+1]]

def paste(dst: Grid, src: Grid, r0: int, c0: int, mode: str = "overwrite") -> Grid:
    out = clone(dst)
    for r in range(len(src)):
        for c in range(len(src[0])):
            rr, cc = r0 + r, c0 + c
            if within(out, rr, cc):
                if mode == "overwrite" or (mode == "nz" and src[r][c] != 0):
                    out[rr][cc] = src[r][c]
    return out

def translate_same_shape(g: Grid, dr: int, dc: int, fill: int = 0) -> Grid:
    h,w = shape(g)
    out = zeros(h,w,fill)
    for r in range(h):
        for c in range(w):
            rr,cc = r - dr, c - dc
            if 0 <= rr < h and 0 <= cc < w:
                out[r][c] = g[rr][cc]
    return out

def scale_nearest(g: Grid, k: int) -> Grid:
    assert k >= 1
    if k == 1: return clone(g)
    h,w = shape(g)
    out = zeros(h*k, w*k, 0)
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            for dr in range(k):
                for dc in range(k):
                    out[r*k+dr][c*k+dc] = v
    return out

def downscale_block_mode(g: Grid, k: int) -> Grid:
    assert k >= 1
    if k == 1: return clone(g)
    H,W = shape(g)
    h,w = H//k, W//k
    out = zeros(h,w,0)
    for br in range(h):
        for bc in range(w):
            cnt = Counter()
            for r in range(br*k,(br+1)*k):
                for c in range(bc*k,(bc+1)*k):
                    cnt[g[r][c]] += 1
            out[br][bc] = argmax_counter(cnt)
    return out

def connected_components(g: Grid, connectivity: int = 4):
    h,w = shape(g)
    seen = [[False]*w for _ in range(h)]
    comps = []
    if connectivity == 4:
        nbrs = [(1,0),(-1,0),(0,1),(0,-1)]
    else:
        nbrs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0 and not seen[r][c]:
                q = [(r,c)]
                seen[r][c] = True
                cur = []
                while q:
                    rr,cc = q.pop()
                    cur.append((rr,cc))
                    for dr,dc in nbrs:
                        r2,c2 = rr+dr, cc+dc
                        if 0 <= r2 < h and 0 <= c2 < w and g[r2][c2] != 0 and not seen[r2][c2]:
                            seen[r2][c2] = True
                            q.append((r2,c2))
                comps.append(cur)
    return comps

def extract_component(g: Grid, comp):
    if not comp: return [[0]]
    rs = [r for r,_ in comp]; cs = [c for _,c in comp]
    rmin,rmax,cmin,cmax = min(rs),max(rs),min(cs),max(cs)
    out = zeros(rmax-rmin+1, cmax-cmin+1, 0)
    for r,c in comp:
        out[r-rmin][c-cmin] = g[r][c]
    return out

def centroid_of_cells(cells):
    cells = list(cells)
    if not cells: return (0.0, 0.0)
    sr = sum(r for r,_ in cells); sc = sum(c for _,c in cells)
    n = len(cells)
    return (sr/n, sc/n)

def apply_colormap(g: Grid, mapping, default: int | None = None) -> Grid:
    out = []
    for row in g:
        out.append([mapping.get(v, default if default is not None else v) for v in row])
    return out

def draw_border(g: Grid, color: int, thickness: int = 1) -> Grid:
    h,w = shape(g)
    out = clone(g)
    for t in range(thickness):
        for c in range(w):
            out[t][c] = color
            out[h-1-t][c] = color
        for r in range(h):
            out[r][t] = color
            out[r][w-1-t] = color
    return out

def tile(src: Grid, out_h: int, out_w: int) -> Grid:
    h,w = shape(src)
    out = zeros(out_h, out_w, 0)
    for r in range(out_h):
        for c in range(out_w):
            out[r][c] = src[r % h][c % w]
    return out

def has_vertical_symmetry(g: Grid) -> bool:
    """Check if grid has vertical (left-right) mirror symmetry"""
    h, w = shape(g)
    for r in range(h):
        for c in range(w // 2):
            if g[r][c] != g[r][w - 1 - c]:
                return False
    return True

def has_horizontal_symmetry(g: Grid) -> bool:
    """Check if grid has horizontal (top-bottom) mirror symmetry"""
    h, w = shape(g)
    for r in range(h // 2):
        for c in range(w):
            if g[r][c] != g[h - 1 - r][c]:
                return False
    return True

def has_rotational_symmetry(g: Grid, order: int = 2) -> bool:
    """Check if grid has rotational symmetry of given order (2 = 180°, 4 = 90°)"""
    if order == 2:
        return eq(g, rot90(g, 2))
    elif order == 4:
        return eq(g, rot90(g, 1))
    return False

def make_symmetric(g: Grid, sym_type: str) -> Grid:
    """Force a grid to be symmetric by copying one half to the other"""
    h, w = shape(g)
    out = clone(g)
    
    if sym_type == "vertical":
        # Copy left half to right half
        for r in range(h):
            for c in range(w // 2):
                out[r][w - 1 - c] = out[r][c]
    elif sym_type == "horizontal":
        # Copy top half to bottom half
        for r in range(h // 2):
            for c in range(w):
                out[h - 1 - r][c] = out[r][c]
    elif sym_type == "both":
        # Make both horizontally and vertically symmetric
        # First make vertically symmetric
        for r in range(h):
            for c in range(w // 2):
                out[r][w - 1 - c] = out[r][c]
        # Then make horizontally symmetric
        for r in range(h // 2):
            for c in range(w):
                out[h - 1 - r][c] = out[r][c]
    
    return out

def fill_holes(g: Grid, background: int = 0) -> Grid:
    """Fill enclosed regions (holes) in the grid with the most common non-background color"""
    h, w = shape(g)
    out = clone(g)
    
    # Find most common non-background color
    color_counts = Counter(v for row in g for v in row if v != background)
    if not color_counts:
        return out
    fill_color = color_counts.most_common(1)[0][0]
    
    # Flood fill from edges to find regions connected to boundary
    visited = [[False] * w for _ in range(h)]
    
    def flood_fill_from_edge(r: int, c: int):
        if not within(g, r, c) or visited[r][c] or g[r][c] != background:
            return
        visited[r][c] = True
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            flood_fill_from_edge(r + dr, c + dc)
    
    # Start flood fill from all edge cells
    for r in range(h):
        flood_fill_from_edge(r, 0)
        flood_fill_from_edge(r, w - 1)
    for c in range(w):
        flood_fill_from_edge(0, c)
        flood_fill_from_edge(h - 1, c)
    
    # Fill unvisited background cells (these are holes)
    for r in range(h):
        for c in range(w):
            if g[r][c] == background and not visited[r][c]:
                out[r][c] = fill_color
    
    return out

def connect_nearest_components(g: Grid, background: int = 0) -> Grid:
    """Connect components by drawing lines between their nearest points"""
    comps = connected_components(g, 4)
    if len(comps) < 2:
        return clone(g)
    
    out = clone(g)
    
    # Find centroids of each component
    centroids = []
    for comp in comps:
        if comp:
            cr, cc = centroid_of_cells(comp)
            centroids.append((int(cr), int(cc)))
    
    # Connect each component to its nearest neighbor
    for i, (r1, c1) in enumerate(centroids):
        min_dist = float('inf')
        nearest = -1
        for j, (r2, c2) in enumerate(centroids):
            if i != j:
                dist = abs(r1 - r2) + abs(c1 - c2)  # Manhattan distance
                if dist < min_dist:
                    min_dist = dist
                    nearest = j
        
        if nearest >= 0:
            r2, c2 = centroids[nearest]
            # Draw line from (r1, c1) to (r2, c2)
            # Use the color of the first component
            if comps[i]:
                color = g[comps[i][0][0]][comps[i][0][1]]
                # Simple line drawing (Bresenham-like)
                dr = 1 if r2 > r1 else -1 if r2 < r1 else 0
                dc = 1 if c2 > c1 else -1 if c2 < c1 else 0
                r, c = r1, c1
                while (r, c) != (r2, c2):
                    if within(out, r, c) and out[r][c] == background:
                        out[r][c] = color
                    if abs(r2 - r) > abs(c2 - c):
                        r += dr
                    else:
                        c += dc
    
    return out
