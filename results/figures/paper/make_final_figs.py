"""Final main-text figure set — minimal-text shapes, legends exported separately.

  fig_mechanism_diagnostic.{png,pdf}  Fig. A  5.1+5.2  (a) EXT vs IMG scatter
                                      (b) VP CAL-EXT sign  (c) CTS CAL-EXT sign
  fig_cts_img_to_cal.{png,pdf}        Fig. B  5.2  CTS IMG->CAL dumbbell, SUV only,
                                      rows grouped detection | segmentation
  fig_vp_cts_alignment.{png,pdf}      Fig. C  5.3  all-cam IMG <-> SUV CTS scatter

  legend_mechanism_row.{png,pdf}      1x4: proj-sampling / extract / det / seg
  legend_mechanism_2x2.{png,pdf}      2x2 variant of the same
  legend_img_cal.{png,pdf}            IMG (open) / CAL (filled)
  legend_suv_bus.{png,pdf}            SUV (solid) / Bus (faded)

No titles / no in-plot legends or commentary. Axis labels + model names only.
Palette: projection-sampling = BLUE #4C72B0, extract-then-place = ORANGE #DD8452.
Numbers verified 2026-06-10 (full-3792 BEVDet/BEVDepth/CAPE; subset768 BEVFormer/
DETR3D/seg; CTS oracle-corrected ratios). SimpleBEV excluded.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = '#4C72B0', '#DD8452'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 100,
})
def grid(ax, axis='both'):
    ax.grid(axis=axis, linestyle=':', color='0.85', linewidth=0.7, zorder=0)
def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, f'{name}.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig); print('wrote', name)
def save_legend(handles, name, ncol):
    fig = plt.figure(figsize=(6, 1))
    leg = fig.legend(handles=handles, loc='center', ncol=ncol, frameon=False,
                     handletextpad=.4, columnspacing=1.4, labelspacing=.4)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bb = bb.expanded(1.06, 1.25)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, f'{name}.{ext}'),
                    dpi=300 if ext == 'png' else None, bbox_inches=bb)
    plt.close(fig); print('wrote', name)

C = lambda mech: BLUE if mech == 'gated' else ORANGE

# ================= shared data (verified) =================
# model, task, mech, mVRS EXT, IMG, CAL  (%)
VP = [
    ('BEVFormer',   'det', 'gated',   83.1, 83.8, 93.5),
    ('DETR3D',      'det', 'gated',   84.2, 83.9, 95.6),
    ('CAPE',        'det', 'extract', 94.0, 83.9, 88.6),
    ('BEVDet',      'det', 'extract', 89.2, 83.0, 87.5),
    ('BEVDepth',    'det', 'extract', 90.2, 82.8, 87.4),
    ('CVT',         'seg', 'extract', 90.9, 83.5, 90.5),
    ('GaussianLSS', 'seg', 'extract', 90.7, 84.5, 92.1),
    ('LaRa',        'seg', 'extract', 91.1, 82.5, 90.0),
    ('LSS',         'seg', 'extract', 88.4, 81.9, 86.3),
    ('PointBeV',    'seg', 'gated',   79.4, 79.6, 92.3),
]
VPd = {v[0]: v for v in VP}
# model, mech, suv EXT, suv CAL, bus EXT, bus CAL, suv IMG, bus IMG (CTS %)
CTS = [
    ('BEVFormer',   'gated',   45.1, 71.9, 25.1, 40.1, 37.2, 18.0),
    ('DETR3D',      'gated',   42.9, 76.9, 30.7, 37.6, 34.9, 26.5),
    ('PointBeV',    'gated',   49.1, 68.4,  4.7, 15.6, 20.2, 11.4),
    ('CAPE',        'extract', 82.3, 34.3, 57.5, 32.9, 33.8, 36.0),
    ('BEVDet',      'extract', 61.5, 16.1, 53.9, 22.8, 17.0,  0.2),
    ('BEVDepth',    'extract', 69.5,  5.7, 29.9, 16.6, 10.3,  0.1),
    ('LSS',         'extract', 69.1, 24.2, 31.4, 17.1, 25.0, 17.1),
    ('GaussianLSS', 'extract', 79.4, 40.6, 37.7, 36.4, 36.1, 14.0),
    ('CVT',         'extract', 43.8, 30.4,  3.4,  1.8, 44.3, 17.7),
    ('LaRa',        'extract', 65.5, 38.5, 22.5, 21.4, 31.0, 27.7),
]
CTSd = {v[0]: v for v in CTS}
# all-camera RRS (0-1) for the 5.3 alignment
# full-3792 (det all 5): BEVFormer .425 / DETR3D .424 / CAPE .400 / BEVDet .360 /
# BEVDepth .325. subset768 (full pending): seg models. Rankings (and rho) unchanged;
# note BEVFormer-DETR3D margin is thin (.4250 vs .4243).
ALLCAM_IMG = {'BEVFormer': .425, 'DETR3D': .424, 'CAPE': .400, 'BEVDet': .360,
              'BEVDepth': .325, 'GaussianLSS': .453, 'CVT': .416, 'LaRa': .378,
              'LSS': .354, 'PointBeV': .289}
MECH = {v[0]: v[2] for v in VP}
TASK = {v[0]: v[1] for v in VP}

# ===== Fig. A : mechanism diagnostic (scatter | VP sign | CTS sign) =====
# (a) is EQUAL-SCALE (xlim=ylim, aspect 1): with unequal scales the y=x line is
# steep and the cluster reads as a linear x-y relation, which is misleading.
fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.1),
                         gridspec_kw={'width_ratios': [1.25, 1, 1], 'wspace': 0.28})
# (a) EXT vs IMG scatter
ax = axes[0]; grid(ax)
LO, HI = 76.5, 97
ax.plot([LO, HI], [LO, HI], color='0.55', ls='--', lw=0.9, zorder=1)
OFFA = {'BEVFormer': (-0.5, 0.85), 'DETR3D': (0, -1.9), 'CAPE': (0.35, 0.35),
        'BEVDet': (-0.55, -0.25), 'BEVDepth': (0.5, -1.9), 'LSS': (-0.5, -0.45),
        'CVT': (0.45, -0.25), 'GaussianLSS': (-1.2, 0.7), 'LaRa': (0.45, -0.2),
        'PointBeV': (0.45, 0.3)}
HAA = {'DETR3D': 'center', 'BEVDepth': 'center', 'BEVDet': 'right', 'LSS': 'right'}
for m, task, mech, e, i, c in VP:
    ax.scatter(e, i, c=C(mech), marker='o' if task == 'det' else '^',
               s=34, edgecolors='k', linewidths=.4, zorder=3)
    dx, dy = OFFA[m]
    ax.annotate(m, (e + dx, i + dy), fontsize=6, ha=HAA.get(m, 'left'))
ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
ax.set_aspect('equal')
ax.set_xlabel('mVRS EXT (%)')
ax.set_ylabel('mVRS IMG (%)')
# (b)+(c): rows in mechanism order (gated 3 | extract 7), shared row alignment
order = [v[0] for v in CTS]
ys = np.arange(len(order))[::-1]
# (b) VP CAL-EXT
ax = axes[1]; grid(ax, 'x')
for y, m in zip(ys, order):
    v = VPd[m]
    g = v[5] - v[3]
    ax.barh(y, g, color=C(v[2]), height=.62, edgecolor='k', linewidth=.35, zorder=3)
    ax.text(-0.5 if g >= 0 else 0.5, y, m, ha='right' if g >= 0 else 'left',
            va='center', fontsize=6.5)
ax.axvline(0, color='k', lw=.9)
ax.axhline(6.5, color='0.55', lw=.8, ls='--')
ax.set_xlabel('mVRS CAL $-$ EXT (%p)')
ax.set_yticks([]); ax.set_xlim(-14, 16); ax.set_ylim(-0.6, 9.6)
ax.spines['left'].set_visible(False)
# (c) CTS CAL-EXT (SUV solid / Bus faded)
ax = axes[2]; grid(ax, 'x')
for y, m in zip(ys, order):
    v = CTSd[m]
    gs, gb = v[3] - v[2], v[5] - v[4]
    ax.barh(y + .18, gs, color=C(v[1]), height=.34, edgecolor='k', linewidth=.35, zorder=3)
    ax.barh(y - .18, gb, color=C(v[1]), height=.34, edgecolor='k', linewidth=.35,
            alpha=.45, zorder=3)
ax.axvline(0, color='k', lw=.9)
ax.axhline(6.5, color='0.55', lw=.8, ls='--')
ax.set_xlabel('CTS CAL $-$ EXT (%p)')
ax.set_yticks([]); ax.set_xticks([-60, -30, 0, 30])
ax.set_xlim(-70, 42); ax.set_ylim(-0.6, 9.6)
ax.spines['left'].set_visible(False)
save(fig, 'fig_mechanism_diagnostic')

# ===== Fig. B : CTS IMG -> CAL dumbbell, SUV only (det | seg rows) =====
DUMB = ['BEVFormer', 'DETR3D', 'CAPE', 'BEVDet', 'BEVDepth',     # detection
        'PointBeV', 'CVT', 'GaussianLSS', 'LSS', 'LaRa']         # segmentation
fig, ax = plt.subplots(figsize=(3.35, 3.0))
grid(ax, 'x')
ys = np.arange(len(DUMB))[::-1]
for y, m in zip(ys, DUMB):
    v = CTSd[m]
    img_v, cal_v = v[6], v[3]
    col = C(v[1])
    ax.plot([img_v, cal_v], [y, y], color=col, lw=1.4, zorder=2)
    ax.scatter([img_v], [y], color='white', edgecolors=col, s=26, zorder=3, linewidths=1.2)
    ax.scatter([cal_v], [y], color=col, edgecolors='k', s=34, zorder=3, linewidths=.35)
ax.axvline(0, color='k', lw=.8)
ax.axhline(4.5, color='0.55', lw=.8, ls='--')          # detection | segmentation
ax.set_xlabel('CTS (%)')
ax.set_xlim(-3, 82)
ax.set_yticks(ys); ax.set_yticklabels(DUMB)
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
save(fig, 'fig_cts_img_to_cal')

# ===== Fig. C : VP all-camera IMG <-> SUV CTS scatter =====
fig, ax = plt.subplots(figsize=(3.35, 3.1))
grid(ax)
suv = {v[0]: v[6] for v in CTS}
OFF2 = {'BEVFormer': (0, 1.2), 'DETR3D': (0, -3.0), 'CAPE': (-.008, -0.2),
        'BEVDet': (.004, .5), 'BEVDepth': (.004, .5), 'GaussianLSS': (0, -3.0),
        'CVT': (.004, .5), 'LaRa': (.004, -.9), 'LSS': (.004, .5), 'PointBeV': (.004, .5)}
HA2 = {'BEVFormer': 'center', 'DETR3D': 'center', 'GaussianLSS': 'center', 'CAPE': 'right'}
for m, ac in ALLCAM_IMG.items():
    ax.scatter(ac, suv[m], c=C(MECH[m]), marker='o' if TASK[m] == 'det' else '^',
               s=42, edgecolors='k', linewidths=.4, zorder=3)
    dx, dy = OFF2[m]
    ax.annotate(m, (ac + dx, suv[m] + dy), fontsize=6.5, ha=HA2.get(m, 'left'))
ax.set_xlabel('VP all-camera IMG (RRS)')
ax.set_ylabel('SUV CTS$_{\\mathrm{IMG}}$ (%)')
ax.set_xlim(0.275, 0.475)
ax.set_ylim(4, 50)
save(fig, 'fig_vp_cts_alignment')

# ===== standalone legends =====
LEG = [plt.Line2D([], [], color=BLUE, marker='s', ls='', ms=7, label='projection-sampling'),
       plt.Line2D([], [], color=ORANGE, marker='s', ls='', ms=7, label='extract-then-place'),
       plt.Line2D([], [], color='k', marker='o', ls='', ms=6, label='detection'),
       plt.Line2D([], [], color='k', marker='^', ls='', ms=6, label='segmentation')]
save_legend(LEG, 'legend_mechanism_row', ncol=4)
save_legend(LEG, 'legend_mechanism_2x2', ncol=2)
save_legend([plt.Line2D([], [], color='0.35', marker='o', mfc='white', ls='', ms=6, label='IMG'),
             plt.Line2D([], [], color='0.35', marker='o', ls='', ms=6, label='CAL')],
            'legend_img_cal', ncol=2)
save_legend([plt.Rectangle((0, 0), 1, 1, fc='0.35', ec='k', lw=.4, label='SUV'),
             plt.Rectangle((0, 0), 1, 1, fc='0.35', ec='k', lw=.4, alpha=.45, label='Bus')],
            'legend_suv_bus', ncol=2)
print('done')
