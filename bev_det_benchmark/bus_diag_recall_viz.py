"""Bus-platform recall diagnosis + BEV pred-vs-GT visualization (token/index aligned).

The earlier viz had a ~10m pred-vs-GT offset because predictions (which carry NO
token) were matched to GT read from the RAW pkl order, while the dataset iterates
its OWN (possibly reordered) data_infos order -> systematic sample mismatch.

Fix: read GT from the SAME dataset object that produced the predictions, indexed by
position (single_gpu_test iterates dataset[i] in data_infos order, so results[i]
corresponds to dataset.data_infos[i]). Pred boxes and ann_infos GT are both in the
LiDAR/ego frame (gt origin lidar2ego z=1.8), so once aligned they compare directly.

Outputs (in out/bus_diag/):
  bus_recall_by_distance.png   -- recall vs ego-distance (+ per-class), bar charts
  bus_bev_examples.png         -- 6 example frames, GT (green=hit/red=miss) vs pred (blue)
  bus_recall_stats.json        -- the numbers behind the plots
"""
import os, sys, json, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']
# nuScenes center-distance match threshold used for the recall proxy (the loosest
# AP threshold; recall here is the detection ceiling at 2.0m center distance).
MATCH_THR = 2.0
SCORE_THR = 0.20          # ignore very-low-score preds for the recall proxy
DIST_BINS = [0, 10, 20, 30, 40, 51.2]


def build_dataset(cfg_path, ann_file, data_root):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    cfg = Config.fromfile(cfg_path)
    cfg.data.test.ann_file = ann_file
    cfg.data.test.data_root = data_root
    cfg.data.test.test_mode = True
    return build_dataset(cfg.data.test)


def pred_arrays(res_i):
    pb = res_i['pts_bbox'] if 'pts_bbox' in res_i else res_i
    boxes = pb['boxes_3d']
    xy = boxes.gravity_center[:, :2].numpy()
    score = pb['scores_3d'].numpy()
    label = pb['labels_3d'].numpy()
    return xy, score, label


def gt_arrays(info):
    gtb, gtl = info['ann_infos']
    gtb = np.asarray(gtb, dtype=np.float32).reshape(-1, 9)
    gtl = np.asarray(gtl, dtype=np.int64).reshape(-1)
    return gtb[:, :2], gtl


def greedy_match(gt_xy, gt_lab, pr_xy, pr_score, pr_lab):
    """Per-class greedy match (highest score first) within MATCH_THR. Returns a
    boolean array 'gt_hit' of len(gt)."""
    gt_hit = np.zeros(len(gt_xy), dtype=bool)
    if len(gt_xy) == 0 or len(pr_xy) == 0:
        return gt_hit
    order = np.argsort(-pr_score)
    taken = gt_hit.copy()
    for p in order:
        if pr_score[p] < SCORE_THR:
            continue
        cand = np.where((~taken) & (gt_lab == pr_lab[p]))[0]
        if len(cand) == 0:
            continue
        d = np.linalg.norm(gt_xy[cand] - pr_xy[p], axis=1)
        j = d.argmin()
        if d[j] <= MATCH_THR:
            taken[cand[j]] = True
    return taken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/bevdet/carla/bevdet-r50-carla.py')
    ap.add_argument('--ann', default='data/bevdet_infos/bus_infos_val.pkl')
    ap.add_argument('--data-root', default='data/nuscenes/')
    ap.add_argument('--results', default='/tmp/bevdet_smoke/bus_results.pkl')
    ap.add_argument('--out', default='/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark/out/bus_diag')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mmcv
    print('[diag] building dataset ...', flush=True)
    ds = build_dataset(args.cfg, args.ann, args.data_root)
    results = mmcv.load(args.results)
    assert len(results) == len(ds.data_infos), (len(results), len(ds.data_infos))
    N = len(results)
    print(f'[diag] {N} samples', flush=True)

    # ---- alignment sanity: median nearest-GT distance for confident preds ----
    near = []
    for i in range(N):
        gt_xy, _ = gt_arrays(ds.data_infos[i])
        pr_xy, pr_s, _ = pred_arrays(results[i])
        m = pr_s >= 0.30
        if gt_xy.size and m.any():
            for p in pr_xy[m]:
                near.append(np.linalg.norm(gt_xy - p, axis=1).min())
    near = np.array(near)
    med = float(np.median(near)) if near.size else float('nan')
    print(f'[diag] ALIGNMENT median nearest-GT dist for score>=0.3 preds = {med:.2f} m '
          f'(>~5m would mean misalignment)', flush=True)

    # ---- per-distance / per-class recall ----
    nbin = len(DIST_BINS) - 1
    tot_d = np.zeros(nbin); hit_d = np.zeros(nbin)
    tot_c = np.zeros(len(CLASSES)); hit_c = np.zeros(len(CLASSES))
    # per-class x per-distance too
    tot_cd = np.zeros((len(CLASSES), nbin)); hit_cd = np.zeros((len(CLASSES), nbin))
    for i in range(N):
        gt_xy, gt_lab = gt_arrays(ds.data_infos[i])
        pr_xy, pr_s, pr_lab = pred_arrays(results[i])
        hit = greedy_match(gt_xy, gt_lab, pr_xy, pr_s, pr_lab)
        if not len(gt_xy):
            continue
        dist = np.linalg.norm(gt_xy, axis=1)
        b = np.clip(np.digitize(dist, DIST_BINS) - 1, 0, nbin - 1)
        for k in range(len(gt_xy)):
            tot_d[b[k]] += 1; hit_d[b[k]] += hit[k]
            c = gt_lab[k]
            if 0 <= c < len(CLASSES):
                tot_c[c] += 1; hit_c[c] += hit[k]
                tot_cd[c, b[k]] += 1; hit_cd[c, b[k]] += hit[k]
    rec_d = np.divide(hit_d, tot_d, out=np.zeros_like(hit_d), where=tot_d > 0)
    rec_c = np.divide(hit_c, tot_c, out=np.zeros_like(hit_c), where=tot_c > 0)
    print('[diag] recall by distance:')
    for k in range(nbin):
        print(f'   {DIST_BINS[k]:>4.0f}-{DIST_BINS[k+1]:>4.0f} m : '
              f'recall={rec_d[k]:.3f}  ({int(hit_d[k])}/{int(tot_d[k])})')
    print('[diag] recall by class:')
    for c in range(len(CLASSES)):
        print(f'   {CLASSES[c]:<12}: recall={rec_c[c]:.3f}  ({int(hit_c[c])}/{int(tot_c[c])})')

    stats = dict(
        match_thr=MATCH_THR, score_thr=SCORE_THR, alignment_median_m=med,
        dist_bins=DIST_BINS,
        recall_by_distance=[dict(lo=DIST_BINS[k], hi=DIST_BINS[k+1],
                                 recall=float(rec_d[k]), hit=int(hit_d[k]), tot=int(tot_d[k]))
                            for k in range(nbin)],
        recall_by_class={CLASSES[c]: dict(recall=float(rec_c[c]), hit=int(hit_c[c]),
                                          tot=int(tot_c[c])) for c in range(len(CLASSES))},
    )
    json.dump(stats, open(os.path.join(args.out, 'bus_recall_stats.json'), 'w'), indent=2)

    # ---------------- recall figure ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = [f'{DIST_BINS[k]:.0f}-{DIST_BINS[k+1]:.0f}' for k in range(nbin)]
    bars = ax[0].bar(labels, rec_d, color='#c0392b')
    for k, b in enumerate(bars):
        ax[0].text(b.get_x()+b.get_width()/2, rec_d[k]+0.02,
                   f'{rec_d[k]:.2f}\n{int(hit_d[k])}/{int(tot_d[k])}',
                   ha='center', va='bottom', fontsize=8)
    ax[0].set_ylim(0, 1.0); ax[0].set_xlabel('ego distance (m)')
    ax[0].set_ylabel(f'recall @ {MATCH_THR:.0f}m center dist')
    ax[0].set_title('BUS platform: recall collapses with distance')
    ax[0].grid(axis='y', alpha=0.3)
    order = np.argsort(-rec_c)
    bars = ax[1].bar([CLASSES[c] for c in order], rec_c[order], color='#2c6fbb')
    for k, c in enumerate(order):
        ax[1].text(k, rec_c[c]+0.02, f'{rec_c[c]:.2f}\n{int(hit_c[c])}/{int(tot_c[c])}',
                   ha='center', va='bottom', fontsize=8)
    ax[1].set_ylim(0, 1.0); ax[1].set_ylabel(f'recall @ {MATCH_THR:.0f}m')
    ax[1].set_title('BUS platform: recall by class'); ax[1].tick_params(axis='x', rotation=30)
    ax[1].grid(axis='y', alpha=0.3)
    fig.suptitle(f'BEVDet bus-viewpoint recall  (align median={med:.2f} m, score>={SCORE_THR})',
                 fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(args.out, 'bus_recall_by_distance.png')
    fig.savefig(p1, dpi=130); plt.close(fig)
    print('[diag] wrote', p1, flush=True)

    # ---------------- BEV example frames ----------------
    # pick 6 frames with the most GT (busy scenes show misses clearly)
    ngt = np.array([len(gt_arrays(ds.data_infos[i])[1]) for i in range(N)])
    pick = np.argsort(-ngt)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    R = 55
    for ax_, i in zip(axes.ravel(), pick):
        gt_xy, gt_lab = gt_arrays(ds.data_infos[i])
        pr_xy, pr_s, pr_lab = pred_arrays(results[i])
        hit = greedy_match(gt_xy, gt_lab, pr_xy, pr_s, pr_lab)
        # ego marker
        ax_.plot(0, 0, 'k^', ms=10)
        ax_.add_patch(plt.Circle((0, 0), 0, fill=False))
        for r in (10, 20, 30, 40, 50):
            ax_.add_patch(plt.Circle((0, 0), r, fill=False, ls=':', ec='gray', alpha=0.4))
        # GT: green if hit, red if missed
        for k in range(len(gt_xy)):
            col = '#27ae60' if hit[k] else '#e74c3c'
            ax_.scatter(gt_xy[k, 1], gt_xy[k, 0], marker='s', s=55,
                        facecolors='none', edgecolors=col, linewidths=1.8)
        # preds (score>=thr): blue x
        m = pr_s >= SCORE_THR
        ax_.scatter(pr_xy[m, 1], pr_xy[m, 0], marker='x', s=40, c='#2c6fbb', linewidths=1.4)
        nmiss = int((~hit).sum())
        ax_.set_title(f'frame {i}: GT={len(gt_xy)} (missed {nmiss}), '
                      f'pred>={SCORE_THR}={int(m.sum())}', fontsize=10)
        ax_.set_xlim(R, -R); ax_.set_ylim(-R, R)   # x-axis = left(+y); forward(+x) up
        ax_.set_aspect('equal'); ax_.set_xlabel('left  +y (m)'); ax_.set_ylabel('forward  +x (m)')
        ax_.grid(alpha=0.2)
    # one shared legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], marker='s', ls='', mfc='none', mec='#27ae60', label='GT hit'),
           Line2D([0],[0], marker='s', ls='', mfc='none', mec='#e74c3c', label='GT missed'),
           Line2D([0],[0], marker='x', ls='', c='#2c6fbb', label=f'pred (score>={SCORE_THR})'),
           Line2D([0],[0], marker='^', ls='', c='k', label='ego (bus)')]
    fig.suptitle('BEVDet bus-viewpoint: BEV prediction vs GT  (ego-frame, aligned, '
                 f'match median={med:.2f} m)', y=0.995, fontsize=13)
    fig.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=4, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = os.path.join(args.out, 'bus_bev_examples.png')
    fig.savefig(p2, dpi=120); plt.close(fig)
    print('[diag] wrote', p2, flush=True)


if __name__ == '__main__':
    main()
