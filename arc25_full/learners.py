
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from collections import Counter
from .grids import Grid, rot90, flip_h, flip_v, transpose, eq, histogram, shape, zeros, crop, bbox_of, connected_components, extract_component, scale_nearest, downscale_block_mode, has_vertical_symmetry, has_horizontal_symmetry, has_rotational_symmetry
from .transforms import Transform, T_identity, T_rot, T_flip_h, T_flip_v, T_transpose, T_recolor, T_translate, T_crop_nz_bbox, T_largest_component, T_smallest_component, T_border, T_scale_up, T_scale_down, T_make_symmetric, T_fill_holes, T_connect_components, compose

def all_same(xs): return len(xs)>0 and all(x==xs[0] for x in xs)

# --- Exact-fit learners that return candidate transforms if they fit all train pairs ---

def learn_rotation(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    ks = []
    for inp,out in pairs:
        ok = False
        for k in range(4):
            if eq(rot90(inp,k), out):
                ks.append(k); ok=True; break
        if not ok: return None
    return T_rot(ks[0]) if all_same(ks) else None

def learn_flip_or_transpose(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    ops = []
    for inp,out in pairs:
        if eq(flip_h(inp), out): ops.append("h")
        elif eq(flip_v(inp), out): ops.append("v")
        elif eq(transpose(inp), out): ops.append("t")
        else: return None
    first = ops[0]
    if all_same(ops):
        return {"h": T_flip_h(), "v": T_flip_v(), "t": T_transpose()}[first]
    return None

def learn_global_recolor(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    mapping = {}
    inv = {}
    for inp,out in pairs:
        if shape(inp)!=shape(out): return None
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                a,b = inp[r][c], out[r][c]
                if a in mapping and mapping[a]!=b: return None
                if b in inv and inv[b]!=a: return None
                mapping[a]=b; inv[b]=a
    return T_recolor(mapping) if mapping else None

def learn_same_shape_translate(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    vecs = []
    for inp,out in pairs:
        if shape(inp)!=shape(out): return None
        h,w = shape(inp)
        found = None
        limit = max(h,w)
        for dr in range(-limit, limit+1):
            for dc in range(-limit, limit+1):
                if eq(out, T_translate(dr,dc).fn(inp)):
                    found=(dr,dc); break
            if found: break
        if not found: return None
        vecs.append(found)
    return T_translate(*vecs[0]) if all_same(vecs) else None

def learn_crop_nz(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    for inp,out in pairs:
        box = bbox_of(inp, nonzero_only=True)
        if not box: return None
        if not eq(crop(inp, box), out): return None
    return T_crop_nz_bbox()

def learn_largest_component(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    for inp,out in pairs:
        comps = T_largest_component().fn(inp)
        if not eq(comps, out): return None
    return T_largest_component()

def learn_smallest_component(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    for inp,out in pairs:
        comps = T_smallest_component().fn(inp)
        if not eq(comps, out): return None
    return T_smallest_component()

def learn_uniform_border(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    # Detect additive border with constant color and thickness 1..3
    for t in [1,2,3]:
        colors = []
        ok = True
        for inp,out in pairs:
            hi,wi = shape(inp); ho,wo = shape(out)
            if ho != hi + 2*t or wo != wi + 2*t: ok=False; break
            # inner crop of out should equal inp
            inner = [row[t:wo-t] for row in out[t:ho-t]]
            if not eq(inner, inp): ok=False; break
            # border color must be constant
            border_vals = []
            for c in range(wo): border_vals.append(out[0][c]); border_vals.append(out[ho-1][c])
            for r in range(ho): border_vals.append(out[r][0]); border_vals.append(out[r][wo-1])
            if len(set(border_vals))!=1: ok=False; break
            colors.append(border_vals[0])
        if ok and all_same(colors):
            return T_border(colors[0], t)
    return None

def learn_integer_scale(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    # exact nearest-neighbor upscale or block-mode downscale with same color counts
    facts = []
    for inp,out in pairs:
        hi,wi = shape(inp); ho,wo = shape(out)
        if ho%hi==0 and wo%wi==0 and ho//hi == wo//wi:
            k = ho//hi
            if eq(scale_nearest(inp,k), out):
                facts.append(("up",k)); continue
        if hi%ho==0 and wi%wo==0 and hi//ho == wi//wo:
            k = hi//ho
            if eq(downscale_block_mode(inp,k), out):
                facts.append(("down",k)); continue
        return None
    if not facts or not all_same(facts): return None
    kind,k = facts[0]
    return T_scale_up(k) if kind=="up" else T_scale_down(k)

def learn_recolor_then_rotation(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    for k in range(4):
        rot_inp = [(rot90(inp,k), out) for inp,out in pairs]
        t = learn_global_recolor(rot_inp)
        if t is not None:
            from .transforms import compose, T_rot
            return compose(t, T_rot(k))
    return None

def learn_rotation_then_recolor(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    from .grids import rot90 as r90
    for k in range(4):
        derot_out = [(inp, r90(out,(4-k)%4)) for inp,out in pairs]
        t = learn_global_recolor(derot_out)
        if t is not None:
            from .transforms import compose, T_rot
            return compose(T_rot(k), t)
    return None

def learn_symmetry_completion(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    """Learn if the task involves making inputs symmetric"""
    # Check if all outputs are symmetric in some way and inputs are not
    sym_types = ["vertical", "horizontal", "both"]
    
    for sym_type in sym_types:
        all_match = True
        for inp, out in pairs:
            # Check if input is not symmetric but output is
            if sym_type == "vertical":
                inp_sym = has_vertical_symmetry(inp)
                out_sym = has_vertical_symmetry(out)
            elif sym_type == "horizontal":
                inp_sym = has_horizontal_symmetry(inp)
                out_sym = has_horizontal_symmetry(out)
            else:  # both
                inp_sym = has_vertical_symmetry(inp) and has_horizontal_symmetry(inp)
                out_sym = has_vertical_symmetry(out) and has_horizontal_symmetry(out)
            
            # Test if making input symmetric produces output
            candidate = T_make_symmetric(sym_type).fn(inp)
            if not eq(candidate, out):
                all_match = False
                break
                
        if all_match:
            return T_make_symmetric(sym_type)
    
    return None

def learn_fill_holes(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    """Learn if the task involves filling holes in the input"""
    for background in [0, 1, 2]:  # Try common background colors
        all_match = True
        for inp, out in pairs:
            candidate = T_fill_holes(background).fn(inp)
            if not eq(candidate, out):
                all_match = False
                break
        if all_match:
            return T_fill_holes(background)
    return None

def learn_connect_components(pairs: List[Tuple[Grid,Grid]]) -> Optional[Transform]:
    """Learn if the task involves connecting components"""
    for background in [0, 1, 2]:  # Try common background colors
        all_match = True
        for inp, out in pairs:
            candidate = T_connect_components(background).fn(inp)
            if not eq(candidate, out):
                all_match = False
                break
        if all_match:
            return T_connect_components(background)
    return None

def collect_exact_fit_learners(pairs: List[Tuple[Grid,Grid]]) -> List[Transform]:
    learners = [
        learn_rotation,
        learn_flip_or_transpose,
        learn_global_recolor,
        learn_same_shape_translate,
        learn_crop_nz,
        learn_largest_component,
        learn_smallest_component,
        learn_uniform_border,
        learn_integer_scale,
        learn_symmetry_completion,
        learn_fill_holes,
        learn_connect_components,
        learn_recolor_then_recolor := learn_recolor_then_rotation,
        learn_rotation_then_recolor,
    ]
    cands = []
    for L in learners:
        try:
            t = L(pairs)  # type: ignore
            if t is not None:
                cands.append(t)
        except Exception:
            continue
    # de-dup by name
    uniq = {}
    for t in cands:
        uniq.setdefault(t.name, t)
    return list(uniq.values())
