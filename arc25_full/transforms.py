
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from .grids import Grid, clone, rot90, flip_h, flip_v, transpose, translate_same_shape, apply_colormap, draw_border, zeros, crop, bbox_of, connected_components, extract_component, scale_nearest, downscale_block_mode, paste, tile, shape, make_symmetric, fill_holes, connect_nearest_components

@dataclass(frozen=True)
class Transform:
    name: str
    fn: Callable[[Grid], Grid]

    def __call__(self, g: Grid) -> Grid:
        return self.fn(g)

def T_identity() -> Transform:
    return Transform("identity", lambda g: clone(g))

def T_rot(k: int) -> Transform:
    return Transform(f"rot{k*90}", lambda g: rot90(g, k))

def T_flip_h() -> Transform:
    return Transform("flip_h", flip_h)

def T_flip_v() -> Transform:
    return Transform("flip_v", flip_v)

def T_transpose() -> Transform:
    return Transform("transpose", transpose)

def T_translate(dr: int, dc: int, fill: int = 0) -> Transform:
    return Transform(f"translate({dr},{dc})", lambda g: translate_same_shape(g, dr, dc, fill))

def T_recolor(mapping: Dict[int,int], default: int | None = None) -> Transform:
    mapping = dict(mapping)
    return Transform(f"recolor{mapping}", lambda g: apply_colormap(g, mapping, default))

def T_border(color: int, thickness: int = 1) -> Transform:
    return Transform(f"border(c={color},t={thickness})", lambda g: draw_border(g, color, thickness))

def T_crop_nz_bbox() -> Transform:
    def _f(g: Grid) -> Grid:
        box = bbox_of(g, nonzero_only=True)
        return crop(g, box) if box else clone(g)
    return Transform("crop_nz_bbox", _f)

def T_largest_component() -> Transform:
    def _f(g: Grid) -> Grid:
        comps = connected_components(g, 4)
        if not comps: return [[0]]
        comps.sort(key=len, reverse=True)
        return extract_component(g, comps[0])
    return Transform("largest_component", _f)

def T_smallest_component() -> Transform:
    def _f(g: Grid) -> Grid:
        comps = connected_components(g, 4)
        if not comps: return [[0]]
        comps.sort(key=len)
        return extract_component(g, comps[0])
    return Transform("smallest_component", _f)

def T_scale_up(k: int) -> Transform:
    return Transform(f"scale_up({k})", lambda g: scale_nearest(g, k))

def T_scale_down(k: int) -> Transform:
    return Transform(f"scale_down({k})", lambda g: downscale_block_mode(g, k))

def T_tile_to(out_h: int, out_w: int) -> Transform:
    return Transform(f"tile({out_h}x{out_w})", lambda g: tile(g, out_h, out_w))

def T_paste_at(src_tf: Transform, r0: int, c0: int, mode: str = "overwrite") -> Transform:
    # Returns a transform that creates an all-zero canvas of the same shape as input, applies src_tf and pastes at offset.
    def _f(g: Grid) -> Grid:
        h,w = shape(g)
        canvas = zeros(h,w,0)
        obj = src_tf.fn(g)
        return paste(canvas, obj, r0, c0, mode=mode)
    return Transform(f"paste({src_tf.name}@{r0},{c0},{mode})", _f)

def T_make_symmetric(sym_type: str) -> Transform:
    return Transform(f"make_symmetric({sym_type})", lambda g: make_symmetric(g, sym_type))

def T_fill_holes(background: int = 0) -> Transform:
    return Transform(f"fill_holes(bg={background})", lambda g: fill_holes(g, background))

def T_connect_components(background: int = 0) -> Transform:
    return Transform(f"connect_components(bg={background})", lambda g: connect_nearest_components(g, background))

def compose(a: Transform, b: Transform) -> Transform:
    return Transform(f"{a.name}∘{b.name}", lambda g: a.fn(b.fn(g)))
