"""Final paper-quality figure set (consolidated, blue/orange palette).

MAIN (3):
  fig_ext_img.{png,pdf}        Fig. A  5.1  EXT vs IMG scatter (1/7 mVRS)
  fig_cal_ext_sign.{png,pdf}   Fig. D  5.1+5.2  CAL-EXT sign bars (VP | CTS)
  fig_vp_cts_align.{png,pdf}   Fig. C  5.3  all-camera IMG <-> SUV CTS alignment
APPENDIX (2):
  fig_app_magnitude.{png,pdf}  magnitude curves (CAL pitch / IMG yaw)
  fig_app_cts_dumbbell.{png,pdf}  CTS IMG->CAL per platform

Palette: projection-sampling = BLUE #4C72B0, extract-then-place = ORANGE #DD8452.
All numbers verified 2026-06-10 (full-3792 for BEVDet/BEVDepth/CAPE; subset768 for
BEVFormer/DETR3D/seg; CTS corrected ratios). SimpleBEV excluded.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREY = '#4C72B0', '#DD8452', '0.45'

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
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
# all-camera RRS (0-1) for 5.3 alignment
ALLCAM_IMG = {'BEVFormer': .425, 'DETR3D': .424, 'CAPE': .400, 'BEVDet': .360,
              'BEVDepth': .325, 'GaussianLSS': .453, 'CVT': .416, 'LaRa': .378,
              'LSS': .354, 'PointBeV': .289}
MECH = {m: v[2] if len(v) == 6 else v[1] for *_, in []} or {}
for v in VP: MECH[v[0]] = v[2]
TASK = {v[0]: v[1] for v in VP}
RHO = {'det': [1.00, -0.62, -0.70], 'seg': [0.90, 0.90, 0.60]}  # verified

LEG = [plt.Line2D([], [], color=BLUE, marker='s', ls='', ms=6, label='projection-sampling'),
       plt.Line2D([], [], color=ORANGE, marker='s', ls='', ms=6, label='extract-then-place'),
       plt.Line2D([], [], color='k', marker='o', ls='', ms=5, label='detection'),
       plt.Line2D([], [], color='k', marker='^', ls='', ms=5, label='segmentation')]

# ================= Fig. A : EXT vs IMG =================
fig, ax = plt.subplots(figsize=(3.35, 3.2))
grid(ax)
LO, HI = 76, 97
ax.plot([LO, HI], [LO, HI], color='0.55', ls='--', lw=0.9, zorder=1)
OFF = {'BEVFormer': (-0.5, 0.75), 'DETR3D': (0, -1.55), 'CAPE': (0.35, 0.3),
       'BEVDepth': (0, -1.55), 'BEVDet': (-0.5, -0.2), 'LSS': (-2.4, -0.4),
       'CVT': (0.45, -0.2), 'GaussianLSS': (-1.1, 0.6), 'LaRa': (0.45, -0.15),
       'PointBeV': (0.4, 0.25)}
HA1 = {'DETR3D': 'center', 'BEVDepth': 'center', 'BEVDet': 'right'}
for m, task, mech, e, i, c in VP:
    ax.scatter(e, i, c=C(mech), marker='o' if task == 'det' else '^',
               s=42, edgecolors='k', linewidths=.4, zorder=3)
    dx, dy = OFF.get(m, (0.35, 0.35))
    ax.annotate(m, (e + dx, i + dy), fontsize=6.5, ha=HA1.get(m, 'left'))
ax.set_xlabel('mVRS EXT (%)')
ax.set_ylabel('mVRS IMG (%)')
ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
ax.set_aspect('equal')
ax.legend(handles=LEG, loc='upper left', framealpha=.95, handletextpad=.3,
          borderpad=.4, labelspacing=.3)
save(fig, 'fig_ext_img')

# ================= Fig. D : CAL-EXT sign (VP | CTS) =================
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))
# (a) VP
ax = axes[0]; grid(ax, 'x')
order = [v for v in VP if v[1] == 'det'] + [v for v in VP if v[1] == 'seg']
ys = np.arange(len(order))[::-1]
for y, v in zip(ys, order):
    g = v[5] - v[3]
    ax.barh(y, g, color=C(v[2]), height=.62, edgecolor='k', linewidth=.35, zorder=3)
    ax.text(-0.5 if g >= 0 else 0.5, y, v[0], ha='right' if g >= 0 else 'left',
            va='center', fontsize=7)
ax.axvline(0, color='k', lw=.9)
ax.axhline(4.5, color='0.55', lw=.8, ls='--')          # det | seg divider
ax.text(-13.4, 4.85, 'detection', fontsize=6.5, color='0.45', va='bottom')
ax.text(-13.4, 4.15, 'segmentation', fontsize=6.5, color='0.45', va='top')
ax.set_xlabel('mVRS CAL $-$ EXT (%p)')
ax.set_title('(a) Viewpoint perturbation', loc='left')
ax.set_yticks([]); ax.set_xlim(-14, 15)
ax.spines['left'].set_visible(False)
# (b) CTS suv/bus
ax = axes[1]; grid(ax, 'x')
ys = np.arange(len(CTS))[::-1]
for y, v in zip(ys, CTS):
    gs, gb = v[3] - v[2], v[5] - v[4]
    ax.barh(y + .18, gs, color=C(v[1]), height=.34, edgecolor='k', linewidth=.35, zorder=3)
    ax.barh(y - .18, gb, color=C(v[1]), height=.34, edgecolor='k', linewidth=.35,
            alpha=.45, zorder=3)
    ax.text(-1.6 if gs >= 0 else 1.6, y, v[0], ha='right' if gs >= 0 else 'left',
            va='center', fontsize=7)
ax.axvline(0, color='k', lw=.9)
ax.axhline(6.5, color='0.55', lw=.8, ls='--')
hs = [plt.Rectangle((0, 0), 1, 1, fc='0.35'), plt.Rectangle((0, 0), 1, 1, fc='0.35', alpha=.45)]
ax.legend(hs, ['SUV', 'Bus'], loc='lower right', framealpha=.95)
ax.set_xlabel('CTS CAL $-$ EXT (%p)')
ax.set_title('(b) Cross-platform transfer', loc='left')
ax.set_yticks([]); ax.set_xlim(-70, 42)
ax.spines['left'].set_visible(False)
save(fig, 'fig_cal_ext_sign')

# ================= Fig. C : VP <-> CTS alignment =================
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={'width_ratios': [1.12, 1], 'wspace': 0.32})
# (a) scatter
ax = axes[0]; grid(ax)
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
ax.set_title('(a) All-camera IMG vs SUV transfer', loc='left')
ax.set_xlim(0.275, 0.475)
ax.set_ylim(4, 50)
ax.text(.97, .04, 'rank corr.  det $\\rho$=+1.00,  seg $\\rho$=+0.90',
        transform=ax.transAxes, fontsize=7, va='bottom', ha='right')
ax.legend(handles=LEG, loc='upper left', framealpha=.95, handletextpad=.3,
          borderpad=.4, labelspacing=.3)
# (b) rho bars
ax = axes[1]; grid(ax, 'y')
labels = ['all-camera\nIMG', 'per-camera\nIMG', 'all-camera\nEXT']
x = np.arange(3); w = .36
ax.bar(x - w/2, RHO['det'], w, color='0.30', edgecolor='k', linewidth=.35,
       label='detection', zorder=3)
ax.bar(x + w/2, RHO['seg'], w, color='0.72', edgecolor='k', linewidth=.35,
       label='segmentation', zorder=3)
for xi, v in zip(x - w/2, RHO['det']):
    ax.text(xi, v + (.06 if v >= 0 else -.13), f'{v:+.2f}', ha='center', fontsize=6.8)
for xi, v in zip(x + w/2, RHO['seg']):
    ax.text(xi, v + (.06 if v >= 0 else -.13), f'{v:+.2f}', ha='center', fontsize=6.8)
ax.axhline(0, color='k', lw=.9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Spearman $\\rho$ vs SUV CTS$_{\\mathrm{IMG}}$')
ax.set_ylim(-1.02, 1.25)
ax.set_title('(b) Which VP component predicts transfer', loc='left')
ax.legend(loc='lower left', framealpha=.95, fontsize=6.5, handlelength=1.1,
          handletextpad=.4, borderpad=.35, labelspacing=.3)
save(fig, 'fig_vp_cts_align')

# ================= Appendix: magnitude curves =================
R = os.path.dirname(HERE)
def co(r): return r.get('condition') or r.get('cond')
def po(r): return r.get('protocol') or r.get('proto')
SRC = {'BEVDet': (f'{R}/BEVDet/vp/eval_vp.json', 'extract', '-'),
       'BEVDepth': (f'{R}/BEVDepth/vp/eval_vp.json', 'extract', '--'),
       'CAPE': (f'{R}/CAPE/vp/vp_cape_sedan_fullframe.json', 'extract', ':'),
       'BEVFormer': (f'{R}/BEVFormer/vp/eval_vp.json', 'gated', '-'),
       'DETR3D': (f'{R}/DETR3D/vp/eval_vp.json', 'gated', '--')}
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
for ax, (cr, axis_, title) in zip(axes, [('CR', 'pitch', '(a) CAL pitch'),
                                         ('VR', 'yaw', '(b) IMG yaw')]):
    grid(ax)
    for m, (f, mech, ls) in SRC.items():
        d = json.load(open(f)); rows = d['rows']
        pts = {r.get('signed_mag', r.get('mag')): r['rrs'] for r in rows
               if co(r) == cr and r['axis'] == axis_ and po(r) == 'all'}
        mags = sorted(pts)
        ax.plot(mags, [pts[k] for k in mags], ls, color=C(mech),
                marker='o', ms=2.6, lw=1.3, label=m, zorder=3)
    ax.axhline(.5, color='0.7', lw=.7, ls=':')
    ax.set_xlabel('signed magnitude (deg)')
    ax.set_title(title, loc='left')
    ax.set_xticks([-20, -12, -4, 4, 12, 20])
axes[0].set_ylabel('all-camera RRS')
axes[0].set_ylim(-0.03, 1.03)
axes[1].legend(loc='lower center', ncol=2, framealpha=.95, fontsize=6.5,
               handlelength=1.6, columnspacing=1.0, borderpad=.35)
save(fig, 'fig_app_magnitude')

# ================= Appendix: CTS IMG->CAL dumbbell =================
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), sharey=True)
order = [v[0] for v in CTS]
for ax, (ii, ci, title) in zip(axes, [(6, 3, '(a) SUV'), (7, 5, '(b) Bus')]):
    grid(ax, 'x')
    ys = np.arange(len(CTS))[::-1]
    for y, v in zip(ys, CTS):
        img_v, cal_v = v[ii], v[ci]
        col = C(v[1])
        ax.plot([img_v, cal_v], [y, y], color=col, lw=1.4, zorder=2)
        ax.scatter([img_v], [y], color='white', edgecolors=col, s=26, zorder=3, linewidths=1.2)
        ax.scatter([cal_v], [y], color=col, edgecolors='k', s=34, zorder=3, linewidths=.35)
        ax.text(-2, y, v[0], ha='right', va='center', fontsize=7)
    ax.axvline(0, color='k', lw=.8)
    ax.axhline(6.5, color='0.55', lw=.8, ls='--')
    ax.set_xlabel('CTS (%)')
    ax.set_title(title, loc='left')
    ax.set_yticks([]); ax.set_xlim(-26, 82)
    ax.spines['left'].set_visible(False)
hs = [plt.Line2D([], [], color='0.35', marker='o', mfc='white', ls='', label='IMG'),
      plt.Line2D([], [], color='0.35', marker='o', ls='', label='CAL')]
axes[1].legend(handles=hs, loc='lower right', framealpha=.95)
save(fig, 'fig_app_cts_dumbbell')
print('done')
