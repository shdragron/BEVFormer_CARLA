"""Section-5 analysis figures (RoboGeo paper flow).

Fig A (5.1): VP EXT vs IMG all-camera correlation scatter — mechanism split.
Fig B (5.2): CTS IMG -> CAL recovery dumbbell — who converts correct extrinsics into transfer.
Fig C (5.3): ranking bump chart NORMAL -> VP-IMG -> VP-CAL -> CTS-IMG — rankings not aligned.

All numbers verified against results/{model}/vp|cts jsons, seg_vp_cts.tsv and paper
Table 2 (3-agent verification pass, 2026-06-10). VP values = all-camera RRS; CTS = Table-2 %.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- data (verified) ----------------
# task, model, mechanism ('gated'|'extract'), EXT, IMG, CAL  (VP all-camera RRS)
VP = [
    ('det', 'BEVFormer',   'gated',   0.428, 0.426, 0.777),
    ('det', 'DETR3D',      'gated',   0.438, 0.422, 0.845),
    ('det', 'CAPE',        'extract', 0.811, 0.407, 0.560),
    ('det', 'BEVDepth',    'extract', 0.652, 0.328, 0.492),
    ('det', 'BEVDet',      'extract', 0.610, 0.364, 0.512),
    ('seg', 'CVT',         'extract', 0.690, 0.416, 0.653),
    ('seg', 'GaussianLSS', 'extract', 0.698, 0.453, 0.699),
    ('seg', 'LaRa',        'extract', 0.703, 0.378, 0.627),
    ('seg', 'LSS',         'extract', 0.619, 0.354, 0.510),
    ('seg', 'PointBeV',    'gated',   0.364, 0.289, 0.756),
    ('seg', 'SimpleBEV',   'outlier', 0.539, 0.242, 0.183),
]

# CTS (%, Table 2): model -> (suv_IMG, suv_CAL, bus_IMG, bus_CAL)
CTS = {
    'BEVDet':      (9.1,  8.6,  0.1,  8.5,  'det', 'extract+depth'),
    'BEVDepth':    (10.3, 5.7,  0.1, 16.6,  'det', 'extract+depth'),
    'BEVFormer':   (37.2, 71.9, 18.0, 40.1, 'det', 'gated'),
    'CAPE':        (33.8, 34.3, 36.0, 32.9, 'det', 'extract'),
    'DETR3D':      (34.9, 76.9, 26.5, 37.6, 'det', 'gated'),
    'LSS':         (25.0, 24.2, 17.1, 17.1, 'seg', 'extract+depth'),
    'GaussianLSS': (36.1, 40.6, 14.0, 36.4, 'seg', 'extract+depth'),
    'CVT':         (44.3, 30.4, 17.7, 1.8,  'seg', 'extract'),
    'SimpleBEV':   (11.8, 13.9, 12.2, 3.4,  'seg', 'outlier'),
    'LaRa':        (31.0, 38.5, 27.7, 21.4, 'seg', 'extract'),
    'PointBeV':    (20.2, 68.4, 11.4, 15.6, 'seg', 'gated'),
}

# NORMAL absolute performance (det NDS / seg IoU)
NORMAL = {
    'CAPE': 0.5508, 'DETR3D': 0.5368, 'BEVDepth': 0.5354, 'BEVDet': 0.5166, 'BEVFormer': 0.5051,
    'SimpleBEV': 0.504, 'GaussianLSS': 0.489, 'PointBeV': 0.481, 'LaRa': 0.454, 'LSS': 0.445, 'CVT': 0.424,
}

COL = {'gated': '#1f77b4', 'extract': '#d62728', 'extract+depth': '#8c1515', 'outlier': '#7f7f7f'}

# ---------------- Fig A: EXT vs IMG scatter (headline 1/7 mVRS, Table-2 scale %) ----------------
# task, model, mech, EXT, IMG (Table 2 mVRS %)
VP17 = [
    ('det', 'BEVFormer',   'gated',   83.1, 83.8),
    ('det', 'DETR3D',      'gated',   84.2, 83.9),
    ('det', 'CAPE',        'extract', 94.3, 84.3),
    ('det', 'BEVDepth',    'extract', 90.5, 83.1),
    ('det', 'BEVDet',      'extract', 89.1, 83.3),
    ('seg', 'CVT',         'extract', 90.9, 83.5),
    ('seg', 'GaussianLSS', 'extract', 90.7, 84.5),
    ('seg', 'LaRa',        'extract', 91.1, 82.5),
    ('seg', 'LSS',         'extract', 88.4, 81.9),
    ('seg', 'PointBeV',    'gated',   79.4, 79.6),
    ('seg', 'SimpleBEV',   'outlier', 75.9, 78.1),
]
fig, ax = plt.subplots(figsize=(5.2, 5.0))
LO, HI = 72, 98
ax.plot([LO, HI], [LO, HI], 'k--', lw=1, alpha=.5, zorder=1)
ax.fill_between([LO, HI], [LO, HI], LO, color='0.93', zorder=0)
ax.text(91.5, 76.5, 'EXT > IMG\n(extrinsic-only\nover-reports robustness)',
        fontsize=8, ha='center', color='0.35')
OFF = {'BEVFormer': (-0.8, 0.9), 'DETR3D': (0.45, -0.9), 'CAPE': (0.4, 0.35),
       'BEVDepth': (0.45, -0.5), 'BEVDet': (-1.0, -1.1), 'LSS': (-2.6, -0.3),
       'CVT': (0.45, -0.25), 'GaussianLSS': (0.3, 0.5), 'LaRa': (0.45, -0.8),
       'PointBeV': (0.4, 0.3), 'SimpleBEV': (0.4, 0.25)}
for task, m, mech, e, i in VP17:
    mk = 'o' if task == 'det' else '^'
    ax.scatter(e, i, c=COL[mech], marker=mk, s=70, zorder=3,
               edgecolors='k', linewidths=.5)
    dx, dy = OFF.get(m, (0.4, 0.4))
    ax.annotate(m, (e + dx, i + dy), fontsize=7.5)
hs = [plt.Line2D([], [], color=COL['gated'], marker='s', ls='', label='sampling-gated'),
      plt.Line2D([], [], color=COL['extract'], marker='s', ls='', label='extract-then-place'),
      plt.Line2D([], [], color=COL['outlier'], marker='s', ls='', label='SimpleBEV (outlier)'),
      plt.Line2D([], [], color='k', marker='o', ls='', label='detection (SDS)'),
      plt.Line2D([], [], color='k', marker='^', ls='', label='segmentation (IoU)')]
ax.legend(handles=hs, fontsize=7.5, loc='upper left', framealpha=.9)
ax.set_xlabel('mVRS EXT (%)')
ax.set_ylabel('mVRS IMG (%)')
ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
ax.set_title('EXT vs IMG: the gap is a mechanism signature', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_vp_ext_img_correlation.png'), dpi=220)
plt.close(fig)

# Spearman EXT-vs-IMG
from scipy import stats as st  # noqa
ext17 = [v[3] for v in VP17]; img17 = [v[4] for v in VP17]
print(f'[FigA] Spearman(EXT, IMG) headline 1/7: {st.spearmanr(ext17, img17).statistic:.2f}')
ext = [v[3] for v in VP]; img = [v[4] for v in VP]
print(f'[FigA] Spearman(EXT, IMG) all-camera : {st.spearmanr(ext, img).statistic:.2f}')

# ---------------- Fig B: CTS IMG -> CAL dumbbell ----------------
order = ['BEVDet', 'BEVDepth', 'BEVFormer', 'CAPE', 'DETR3D',
         'LSS', 'GaussianLSS', 'CVT', 'SimpleBEV', 'LaRa', 'PointBeV']
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
for ax, (i_idx, c_idx, title) in zip(axes, [(0, 1, 'SUV'), (2, 3, 'Bus')]):
    ys = np.arange(len(order))[::-1]
    for y, m in zip(ys, order):
        v = CTS[m]; img_v, cal_v = v[i_idx], v[c_idx]
        mech = v[5]
        col = '#2ca02c' if cal_v >= img_v else '#d62728'
        ax.annotate('', xy=(cal_v, y), xytext=(img_v, y),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8))
        ax.scatter([img_v], [y], color='0.4', s=28, zorder=3)
        ax.scatter([cal_v], [y], color=col, s=40, zorder=3, edgecolors='k', linewidths=.4)
        ax.text(-3.5, y, m, ha='right', va='center', fontsize=8,
                color=COL.get(mech, 'k') if mech != 'extract+depth' else COL['extract+depth'])
    ax.axhline(5.5, color='0.8', lw=.8)
    ax.text(0.985, 0.02, 'detection (top) / segmentation (bottom)',
            transform=ax.transAxes, fontsize=6.5, ha='right', color='0.5')
    ax.set_title(f'{title}: CTS IMG $\\rightarrow$ CAL', fontsize=10)
    ax.set_xlabel('CTS (%)')
    ax.set_xlim(-30, 85)
    ax.set_yticks([])
axes[0].text(60, 7.7, 'gray dot = IMG\narrow head = CAL\ngreen = extrinsics help\nred = extrinsics hurt',
             fontsize=7, va='top', color='0.35',
             bbox=dict(fc='white', ec='0.8', alpha=.9))
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_cts_img_to_cal.png'), dpi=220)
plt.close(fig)

# ---------------- Fig C: ranking bump chart ----------------
def ranks(d):  # higher value = better -> rank 1 best
    items = sorted(d.items(), key=lambda kv: -kv[1])
    return {m: r + 1 for r, (m, _) in enumerate(items)}

det_models = ['CAPE', 'DETR3D', 'BEVDepth', 'BEVDet', 'BEVFormer']
seg_models = ['SimpleBEV', 'GaussianLSS', 'PointBeV', 'LaRa', 'LSS', 'CVT']
vp_img = {m: i for _, m, _, e, i, c in VP}
vp_cal = {m: c for _, m, _, e, i, c in VP}
cts_img = {m: (CTS[m][0] + CTS[m][2]) / 2 for m in CTS}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for ax, models, title in [(axes[0], det_models, 'Detection (SDS)'),
                          (axes[1], seg_models, 'Segmentation (IoU)')]:
    cols = ['NORMAL', 'VP-IMG', 'VP-CAL', 'CTS-IMG']
    rk = [ranks({m: NORMAL[m] for m in models}),
          ranks({m: vp_img[m] for m in models}),
          ranks({m: vp_cal[m] for m in models}),
          ranks({m: cts_img[m] for m in models})]
    for m in models:
        mech = next(v[2] for v in VP if v[1] == m)
        ys = [r[m] for r in rk]
        ax.plot(range(4), ys, '-o', color=COL[mech], lw=1.6, ms=5)
        ax.text(-0.12, ys[0], m, ha='right', va='center', fontsize=8)
        ax.text(3.12, ys[-1], m, ha='left', va='center', fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=8.5)
    ax.set_ylim(len(models) + .5, .5)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_ylabel('rank (1 = best)')
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1.1, 4.1)
    ax.grid(axis='y', alpha=.25)
    # Spearman vs NORMAL
    base = [rk[0][m] for m in models]
    for j, name in enumerate(cols[1:], 1):
        cur = [rk[j][m] for m in models]
        rho = st.spearmanr(base, cur).statistic
        ax.text(j, len(models) + .35, f'$\\rho$={rho:.2f}', ha='center', fontsize=7.5, color='0.4')
fig.suptitle('In-distribution ranking does not predict camera-geometry robustness', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, 'fig_ranking_bump.png'), dpi=220)
plt.close(fig)

# print rank tables + spearman for the text
for name, models in [('det', det_models), ('seg', seg_models)]:
    rk0 = ranks({m: NORMAL[m] for m in models})
    rki = ranks({m: vp_img[m] for m in models})
    rkc = ranks({m: vp_cal[m] for m in models})
    rkt = ranks({m: cts_img[m] for m in models})
    base = [rk0[m] for m in models]
    print(f'[{name}] spearman NORMAL vs VP-IMG  {st.spearmanr(base,[rki[m] for m in models]).statistic:.2f}')
    print(f'[{name}] spearman NORMAL vs VP-CAL  {st.spearmanr(base,[rkc[m] for m in models]).statistic:.2f}')
    print(f'[{name}] spearman NORMAL vs CTS-IMG {st.spearmanr(base,[rkt[m] for m in models]).statistic:.2f}')
    print(f'[{name}] spearman VP-IMG vs CTS-IMG {st.spearmanr([rki[m] for m in models],[rkt[m] for m in models]).statistic:.2f}')
print('figures written to', HERE)
