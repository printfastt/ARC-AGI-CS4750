
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import json
from pathlib import Path
from collections import Counter
from .grids import Grid, shape, clone, zeros, histogram
from .transforms import Transform
from .learners import collect_exact_fit_learners
from .search import fit_on_pairs

@dataclass
class Task:
    task_id: str
    train_pairs: List[Tuple[Grid,Grid]]
    test_inputs: List[Grid]

def load_task(path: str, task_id: Optional[str]=None) -> Task:
    js = json.loads(Path(path).read_text())
    train = [(p["input"], p["output"]) for p in js["train"]]
    tests = [p["input"] for p in js["test"]]
    return Task(task_id or Path(path).stem, train, tests)

def discover_tasks(eval_dir: str) -> List[Task]:
    tasks = []
    p = Path(eval_dir)
    for q in sorted(p.glob("*.json")):
        tasks.append(load_task(str(q), q.stem))
    for q in sorted(p.glob("*/*.json")):
        tasks.append(load_task(str(q), q.stem))
    return tasks

def rank_transforms(cands: List[Transform]) -> List[Transform]:
    def priority_score(t: Transform) -> Tuple[int, int, int, str]:
        """Return (priority_class, composition_count, name_length, name) for sorting"""
        name = t.name
        
        # Priority 0: High-value exact matches (most likely to be correct)
        if any(pattern in name for pattern in ["recolor", "rot", "flip", "transpose"]):
            return (0, name.count("∘"), len(name), name)
        
        # Priority 1: Geometric and structural transforms
        if any(pattern in name for pattern in ["crop", "scale", "component", "symmetric"]):
            return (1, name.count("∘"), len(name), name)
        
        # Priority 2: Pattern-based transforms  
        if any(pattern in name for pattern in ["fill_holes", "connect_components", "border"]):
            return (2, name.count("∘"), len(name), name)
        
        # Priority 3: Spatial transforms
        if "translate" in name:
            return (3, name.count("∘"), len(name), name)
        
        # Priority 4: Everything else
        return (4, name.count("∘"), len(name), name)
    
    return sorted(cands, key=priority_score)

def predict_with_transforms(tfs: List[Transform], test_inp: Grid) -> List[Grid]:
    outs = []
    for tf in tfs:
        try:
            outs.append(tf.fn(test_inp))
        except Exception:
            continue
    return outs

def fallback_attempts(task: Task, test_inp: Grid) -> List[Grid]:
    a = clone(test_inp)  # identity
    # fill with most common train OUT color if available, else most common in test
    out_colors = Counter(v for _,out in task.train_pairs for row in out for v in row)
    if out_colors:
        mode = out_colors.most_common(1)[0][0]
    else:
        mode = histogram(test_inp).most_common(1)[0][0] if test_inp else 0
    b = zeros(len(test_inp), len(test_inp[0]), mode)
    return [a,b]

def solve_task(task: Task) -> List[Tuple[Grid,Grid]]:
    pairs = task.train_pairs

    # 1) exact-fit analytical learners
    cands = collect_exact_fit_learners(pairs)

    # 2) short program search as a backstop
    search_cands = fit_on_pairs(pairs, max_len=4, limit=3000)
    cands += search_cands

    ordered = rank_transforms(cands)

    results: List[Tuple[Grid,Grid]] = []
    for test_inp in task.test_inputs:
        preds = predict_with_transforms(ordered, test_inp)

        # ensure two distinct grids
        uniq = []
        seen = set()
        for g in preds:
            key = tuple(v for row in g for v in row)
            if key not in seen:
                uniq.append(g); seen.add(key)
            if len(uniq) >= 2: break
        if len(uniq) < 2:
            uniq += fallback_attempts(task, test_inp)
            uniq = uniq[:2]
        results.append((uniq[0], uniq[1]))
    return results

def to_submission_record(task_id: str, attempts: List[Tuple[Grid,Grid]]) -> Dict[str, Any]:
    return {task_id: [ {"attempt_1": a1, "attempt_2": a2} for (a1,a2) in attempts ]}

def solve_all_tasks(eval_dir: str, limit: Optional[int] = None) -> Dict[str, Any]:
    tasks = discover_tasks(eval_dir)
    if limit is not None:
        tasks = tasks[:limit]
    sub = {}
    for i,task in enumerate(tasks,1):
        attempts = solve_task(task)
        sub.update(to_submission_record(task.task_id, attempts))
        print(f"[{i}/{len(tasks)}] solved {task.task_id} ({len(attempts)} tests).")
    return sub

def generate_submission(eval_dir: str, out_path: str = "submission.json", limit: Optional[int] = None) -> str:
    sub = solve_all_tasks(eval_dir, limit=limit)
    with open(out_path, "w") as f:
        json.dump(sub, f)
    print(f"Wrote {out_path} with {len(sub)} tasks.")
    return out_path
