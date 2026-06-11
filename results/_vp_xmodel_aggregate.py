#!/usr/bin/env python3
"""Aggregate per-config VP CSVs into a canonical axis-resolved cross-model RRS table.

Canonical cell = mean RRS over the 5 magnitudes (±[4,8,12,16,20]) for a given
(condition, axis, scope). per-cam = also averaged over the 6 cameras; all-cam =
protocol 'all'. ALL-axis row = mean over {roll,pitch,yaw}. mVRS = mean(per-cam, all-cam).
Pure arithmetic, deterministic. Emits markdown + JSON ground truth.
"""
import csv, json, os, collections

RESULTS = os.path.dirname(os.path.abspath(__file__))
MODELS = ['BEVDet', 'BEVDepth', 'BEVFormer']
CONDS = ['ER', 'VR', 'CR']           # EXT / IMG(primary) / CAL
AXES = ['roll', 'pitch', 'yaw']
# BEVDet/BEVDepth per_config CSVs omit the Normal row; P_NORMAL taken from their
# eval_vp_summary.txt. BEVFormer's CSV carries its own Normal row (overrides below).
# Cross-model table stays MATCHED at the 768 subset for every model (per-model
# eval_vp.* may be the full-3792 run; full≈subset, |ΔRRS| ≤0.01 — confirmed for
# BEVDet/BEVDepth/BEVFormer-allcam). load() prefers the *.subset768.csv backup
# when the canonical CSV has been promoted to full.
P_NORMAL = {
    'BEVDet':    {'nds': 0.5185, 'map6': 0.4695},   # 768 subset
    'BEVDepth':  {'nds': 0.5324, 'map6': 0.4931},   # 768 subset
    'BEVFormer': {'nds': 0.5051, 'map6': 0.4449},   # 768 subset (frame-fixed)
}

def load(model):
    sub = os.path.join(RESULTS, model, 'vp', 'eval_vp_per_config.subset768.csv')
    path = sub if os.path.exists(sub) else \
        os.path.join(RESULTS, model, 'vp', 'eval_vp_per_config.csv')
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

# accum[model][cond][axis]['percam'|'allcam'] -> list of rrs
def aggregate(model, rows):
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: {'percam': [], 'allcam': []}))
    for r in rows:
        cond = r['condition']
        if cond == 'Normal':
            P_NORMAL[model] = {'nds': float(r['nds']), 'map6': float(r['map6'])}
            continue
        if cond not in CONDS:
            continue
        axis = r['axis']
        if axis not in AXES:
            continue
        scope = r['protocol']
        rrs = float(r['rrs'])
        if scope == 'all':
            acc[cond][axis]['allcam'].append(rrs)
        elif scope.startswith('CAM_'):
            acc[cond][axis]['percam'].append(rrs)
    return acc

def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')

summary = {}
for model in MODELS:
    rows = load(model)
    acc = aggregate(model, rows)
    summary[model] = {'P_NORMAL': P_NORMAL.get(model, {}), 'table': {}}
    for cond in CONDS:
        summary[model]['table'][cond] = {}
        pc_axis, ac_axis = [], []
        for axis in AXES:
            pc = mean(acc[cond][axis]['percam'])
            ac = mean(acc[cond][axis]['allcam'])
            summary[model]['table'][cond][axis] = {'percam': pc, 'allcam': ac,
                                                    'n_percam': len(acc[cond][axis]['percam']),
                                                    'n_allcam': len(acc[cond][axis]['allcam'])}
            pc_axis.append(pc); ac_axis.append(ac)
        pc_all = mean(pc_axis); ac_all = mean(ac_axis)
        summary[model]['table'][cond]['ALL'] = {'percam': pc_all, 'allcam': ac_all,
                                                 'mVRS': (pc_all + ac_all) / 2}

# ---- markdown ----
def f3(x): return f"{x:.3f}"
print("# VP cross-model — canonical axis-resolved RRS (mean over 5 magnitudes)\n")
print("Conditions: ER=EXT(extrinsic), VR=IMG(image/viewpoint, **primary**), CR=CAL(both, consistent).")
print("per-cam = mean over 6 cams×5 mags; all-cam = mean over 5 mags. ALL = mean over roll/pitch/yaw.\n")
for model in MODELS:
    pn = summary[model]['P_NORMAL']
    print(f"## {model}   P_NORMAL NDS={f3(pn['nds'])}  mAP6={f3(pn['map6'])}\n")
    print("| cond | scope | roll | pitch | yaw | ALL |")
    print("|---|---|---|---|---|---|")
    for cond in CONDS:
        t = summary[model]['table'][cond]
        prim = " (primary)" if cond == 'VR' else ""
        print(f"| {cond}{prim} | per-cam | {f3(t['roll']['percam'])} | {f3(t['pitch']['percam'])} | {f3(t['yaw']['percam'])} | {f3(t['ALL']['percam'])} |")
        print(f"| {cond}{prim} | all-cam | {f3(t['roll']['allcam'])} | **{f3(t['pitch']['allcam'])}** | {f3(t['yaw']['allcam'])} | {f3(t['ALL']['allcam'])} |")
    print()

# ---- cross-model headline pivots ----
print("## Cross-model headline (all-cam, axis-mean ALL)\n")
print("| cond | BEVDet | BEVDepth | BEVFormer |")
print("|---|---|---|---|")
for cond in CONDS:
    cells = " | ".join(f3(summary[m]['table'][cond]['ALL']['allcam']) for m in MODELS)
    print(f"| {cond} all-cam | {cells} |")
print()
print("## Cross-model all-cam PITCH (worst axis)\n")
print("| cond | BEVDet | BEVDepth | BEVFormer |")
print("|---|---|---|---|")
for cond in CONDS:
    cells = " | ".join(f3(summary[m]['table'][cond]['pitch']['allcam']) for m in MODELS)
    print(f"| {cond} all-cam pitch | {cells} |")

with open(os.path.join(RESULTS, '_vp_xmodel_ground_truth.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("\n[wrote _vp_xmodel_ground_truth.json]")
