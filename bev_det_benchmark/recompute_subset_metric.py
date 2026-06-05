"""Recompute mAP / NDS over an ARBITRARY class subset from a saved detail dict.

Both drivers persist the FULL per-class detail dict (per-class AP at every dist
threshold + per-class TP errors) for every cell:
  * CTS : eval_cts.json  -> rows[].metrics
  * VP  : eval_vp.json   -> rows[].metrics  (+ normal_metrics)
so any subset metric (e.g. the 5-class score) can be recomputed offline WITHOUT
re-running inference. The formula is exactly CarlaNuScenesDataset._evaluate_single's
6-class recompute (nuScenes devkit):
    mAP_S        = mean_{c in S} mean_{dist th} AP[c, th]
    TPscore_k    = max(0, 1 - nanmean_{c in S} TP[c, k])      (k = the 5 TP errors)
    NDS_S        = (5*mAP_S + sum_k TPscore_k) / (5 + 5)

Usage:
    # one detail dict (e.g. printed [CARLA-METRICS-JSON]) as a json file/string
    python recompute_subset_metric.py detail.json \
        --classes car truck bus motorcycle pedestrian
    # a whole CTS/VP results json -> recompute the subset metric for every row
    python recompute_subset_metric.py out/cts_bevdet_sedan/eval_cts.json \
        --classes car truck bus motorcycle pedestrian
"""
import argparse
import json

import numpy as np

PREFIX = 'pts_bbox_NuScenes/'
TP_KEYS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
ALL6 = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']


def subset_map_nds(detail, classes, prefix=PREFIX):
    """(mAP, NDS) over `classes`, recomputed from a saved detail dict."""
    per_class_ap = []
    for c in classes:
        aps = [v for k, v in detail.items()
               if k.startswith(f'{prefix}{c}_AP_dist_')]
        if not aps:
            raise KeyError(f'no AP entries for class {c!r} (prefix {prefix})')
        per_class_ap.append(float(np.mean(aps)))
    mAP = float(np.mean(per_class_ap))
    tp_scores = []
    for k in TP_KEYS:
        vals = [detail[f'{prefix}{c}_{k}'] for c in classes
                if f'{prefix}{c}_{k}' in detail]
        tp_scores.append(max(0.0, 1.0 - float(np.nanmean(vals))))
    nds = (5.0 * mAP + float(np.sum(tp_scores))) / (5.0 + len(TP_KEYS))
    return mAP, nds


def _label(row):
    if 'platform' in row and 'condition' in row:          # CTS row
        return f"{row['platform']:<7} {row['condition']}"
    if 'cond' in row:                                     # VP row
        return f"{row['cond']} {row.get('axis')}{row.get('mag'):+d} {row.get('proto')}"
    return '(detail)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json', help='a detail dict, or a CTS/VP results json')
    ap.add_argument('--classes', nargs='+', default=ALL6)
    ap.add_argument('--prefix', default=PREFIX)
    args = ap.parse_args()
    data = json.load(open(args.json))
    print(f'subset = {args.classes}')

    # whole results json (CTS/VP) -> per-row; else treat as a single detail dict
    rows = data.get('rows') if isinstance(data, dict) else None
    if rows is not None:
        if data.get('normal_metrics'):                    # VP oracle row
            m, n = subset_map_nds(data['normal_metrics'], args.classes, args.prefix)
            print(f'  {"Normal/oracle":<28} mAP={m:.4f} NDS={n:.4f}')
        for r in rows:
            det = r.get('metrics')
            if not det:
                continue
            m, n = subset_map_nds(det, args.classes, args.prefix)
            print(f'  {_label(r):<28} mAP={m:.4f} NDS={n:.4f}')
    else:
        m, n = subset_map_nds(data, args.classes, args.prefix)
        print(f'  mAP={m:.4f} NDS={n:.4f}')


if __name__ == '__main__':
    main()
