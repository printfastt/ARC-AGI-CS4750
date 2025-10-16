
from __future__ import annotations
from typing import List, Tuple
from .grids import Grid, shape
from .transforms import Transform, compose, T_identity, T_rot, T_flip_h, T_flip_v, T_transpose, T_crop_nz_bbox, T_largest_component, T_smallest_component, T_scale_up, T_scale_down, T_border, T_translate, T_recolor, T_make_symmetric, T_fill_holes, T_connect_components

# A small library of search primitives with few parameter choices
def library() -> List[Transform]:
    basics = [
        T_identity(),
        T_rot(1), T_rot(2), T_rot(3),
        T_flip_h(), T_flip_v(), T_transpose(),
        T_crop_nz_bbox(),
        T_largest_component(), T_smallest_component(),
    ]
    # symmetry and pattern transforms
    patterns = [
        T_make_symmetric("vertical"),
        T_make_symmetric("horizontal"),
        T_make_symmetric("both"),
        T_fill_holes(0),
        T_connect_components(0),
    ]
    # small set of translations and borders
    trans = [T_translate(dr,dc) for dr in (-2,-1,0,1,2) for dc in (-2,-1,0,1,2) if not (dr==0 and dc==0)]
    borders = [T_border(c,1) for c in range(1,10)]
    return basics + patterns + trans + borders

def compose_up_to_len(max_len: int) -> List[List[Transform]]:
    # return sequences of transforms of length 1..max_len
    lib = library()
    seqs: List[List[Transform]] = []
    for t in lib:
        seqs.append([t])
    if max_len >= 2:
        for a in lib:
            for b in lib:
                seqs.append([a,b])
    if max_len >= 3:
        for a in lib:
            for b in lib:
                for c in lib:
                    seqs.append([a,b,c])
    return seqs

def run_sequence(seq: List[Transform], g: Grid) -> Grid:
    out = g
    for t in seq:
        out = t.fn(out)
    return out

def fit_on_pairs(pairs: List[Tuple[Grid,Grid]], max_len: int = 2, limit: int = 2000) -> List[Transform]:
    """Enumerate short programs and keep those that exactly solve all train pairs."""
    seqs = compose_up_to_len(max_len)
    if limit is not None:
        seqs = seqs[:limit]
    valids: List[Transform] = []
    seen = set()
    for seq in seqs:
        ok = True
        for inp,out in pairs:
            try:
                pred = run_sequence(seq, inp)
            except Exception:
                ok = False; break
            if pred != out:
                ok = False; break
        if ok:
            # compose into a single Transform for easy use
            from .transforms import compose
            tf = seq[0]
            for s in seq[1:]:
                tf = compose(tf, s)
            if tf.name not in seen:
                valids.append(tf); seen.add(tf.name)
    return valids
