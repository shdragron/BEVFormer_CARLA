"""Single-figure candidate: VP EXT -> IMG ranking-inversion bump chart (1/7 mVRS).

fig_ranking_bump.{png,pdf} — rank trajectories EXT -> IMG (headline 1/7 mVRS, %),
detection (left) and segmentation (right) panels. Values printed next to each node
because the detection IMG column is compressed (82.8-83.9, 1.1pt spread) — ranks
alone would overstate the differences there.

The point (section 5.1): extrinsic-only evaluation changes WHICH models look robust.
On EXT the extract-then-place detectors lead (CAPE 94.0, BEVDepth 90.2, BEVDet 89.2)
while projection-sampling sits at the bottom; under re-rendered IMG every detector
lands in the same 1-point band and the order reshuffles (BEVDepth 2nd -> 5th,
DETR3D 4th -> 2nd, BEVFormer 5th -> 3rd).

Data (previously verified, full-3792 for BEVDet/BEVDepth/CAPE, subset768 for
BEVFormer/DETR3D/seg — make_final_figs.py VP table). Rank tiebreak for the det IMG
near-tie uses the precise values: CAPE 83.90 > DETR3D 83.88 > BEVFormer 83.81
(CAPE/vp/vp_cape_sedan_fullframe_summary.txt, DETR3D/vp/eval_vp_summary.txt,
_vp_xmodel_ground_truth.json).
Palette: projection-sampling = BLUE #4C72B0, extract-then-place = ORANGE #DD8452.
No titles / no in-plot legends (use legend_mechanism_row); panel tags + model names only.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = '#4C72B0', '#DD8452'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 100,
})

MECH = {'BEVFormer': 'gated', 'DETR3D': 'gated', 'PointBeV': 'gated',
        'CAPE': 'extract', 'BEVDet': 'extract', 'BEVDepth': 'extract',
        'CVT': 'extract', 'GaussianLSS': 'extract', 'LaRa': 'extract', 'LSS': 'extract'}
C = lambda m: BLUE if MECH[m] == 'gated' else ORANGE

# headline 1/7 mVRS (%); IMG carries extra digits only to order the det near-tie
EXT = {'CAPE': 94.0, 'BEVDepth': 90.2, 'BEVDet': 89.2, 'DETR3D': 84.2,
       'BEVFormer': 83.1, 'LaRa': 91.1, 'CVT': 90.9, 'GaussianLSS': 90.7,
       'LSS': 88.4, 'PointBeV': 79.4}
IMG = {'CAPE': 83.90, 'DETR3D': 83.88, 'BEVFormer': 83.81, 'BEVDet': 83.0,
       'BEVDepth': 82.8, 'GaussianLSS': 84.5, 'CVT': 83.5, 'LaRa': 82.5,
       'LSS': 81.9, 'PointBeV': 79.6}

DET = ['CAPE', 'BEVDepth', 'BEVDet', 'DETR3D', 'BEVFormer']
SEG = ['LaRa', 'CVT', 'GaussianLSS', 'LSS', 'PointBeV']
COLS = [('EXT', EXT), ('IMG', IMG)]


def ranks(d, models):  # higher = better -> rank 1 best
    order = sorted(models, key=lambda m: -d[m])
    return {m: r + 1 for r, m in enumerate(order)}


fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45))
for ax, models, mark, tag in [(axes[0], DET, 'o', 'Detection'),
                              (axes[1], SEG, '^', 'Segmentation')]:
    rk = [ranks(d, models) for _, d in COLS]
    for m in models:
        ys = [r[m] for r in rk]
        ax.plot(range(len(COLS)), ys, '-', color=C(m), lw=1.5, zorder=2,
                solid_capstyle='round')
        ax.plot(range(len(COLS)), ys, mark, color=C(m), ms=4.5, zorder=3,
                markeredgecolor='k', markeredgewidth=.4)
        ax.text(-0.10, ys[0], f'{m}  {EXT[m]:.1f}', ha='right', va='center',
                fontsize=6.5)
        ax.text(1.10, ys[-1], f'{IMG[m]:.1f}  {m}', ha='left', va='center',
                fontsize=6.5)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([c for c, _ in COLS])
    ax.set_ylim(len(models) + 0.45, 0.55)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_xlim(-1.95, 2.95)
    ax.grid(axis='y', linestyle=':', color='0.85', linewidth=0.7, zorder=0)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.text(0.0, 1.02, tag, transform=ax.transAxes, fontsize=7, color='0.45',
            ha='left', va='bottom')
axes[0].set_ylabel('rank (1 = best)')
axes[1].set_yticklabels([])
fig.tight_layout(w_pad=2.0)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'fig_ranking_bump.{ext}'),
                dpi=300 if ext == 'png' else None, bbox_inches='tight')
plt.close(fig)
print('wrote fig_ranking_bump')

# rank tables + Spearman for the caption
from scipy import stats as st
for name, models in [('det', DET), ('seg', SEG)]:
    rk = [ranks(d, models) for _, d in COLS]
    print(f'[{name}]')
    for m in models:
        print('  ', m, [r[m] for r in rk])
    ext_r = [rk[0][m] for m in models]
    img_r = [rk[1][m] for m in models]
    print(f'   rho(EXT, IMG) = {st.spearmanr(ext_r, img_r).statistic:+.2f}')
