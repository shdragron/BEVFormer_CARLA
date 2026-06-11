"""Section-5 figures, part 2 (final v3 analysis).

Fig D  fig_cal_ext_sign.png      : CAL-EXT gap sign per model — the diagnostic,
                                   panel (a) VP (mVRS pts), panel (b) CTS (pts, suv/bus).
Fig Y  fig_magnitude_curves.png  : all-camera RRS vs signed magnitude (appendix),
                                   panel (a) CAL pitch (mechanism discriminator),
                                   panel (b) IMG yaw (mechanism-independent knee).

VP mVRS values = final Table-2 numbers (full for BEVDet/BEVDepth/CAPE, subset768 for
BEVFormer/DETR3D, subset for seg). CTS = ratio vs platform oracle (BEVDet corrected).
Magnitude curves computed from result jsons (all-camera rrs rows).
SimpleBEV excluded everywhere.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, glob, os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)

CB, CR_, CG = '#1f77b4', '#d62728', '0.45'   # gated blue, extract red

# ---------------- Fig D: CAL-EXT sign ----------------
# (model, task, mech, mVRS_EXT, mVRS_CAL)
VP = [
    ('BEVFormer',   'det', 'gated',   83.1, 93.5),
    ('DETR3D',      'det', 'gated',   84.2, 95.6),
    ('PointBeV',    'seg', 'gated',   79.4, 92.3),
    ('CAPE',        'det', 'extract', 94.0, 88.6),
    ('BEVDet',      'det', 'extract', 89.2, 87.5),
    ('BEVDepth',    'det', 'extract', 90.2, 87.4),
    ('LSS',         'seg', 'extract', 88.4, 86.3),
    ('GaussianLSS', 'seg', 'extract', 90.7, 92.1),   # sole exception (borderline all-cam)
    ('CVT',         'seg', 'extract', 90.9, 90.5),
    ('LaRa',        'seg', 'extract', 91.1, 90.0),
]
# CTS EXT/CAL (%): detectors from results csv ratios; seg = raw IoU / Table-2 oracle
CTS = [
    # (model, mech, suv_EXT, suv_CAL, bus_EXT, bus_CAL)
    ('BEVFormer',   'gated',   45.1, 71.9, 25.1, 40.1),
    ('DETR3D',      'gated',   42.9, 76.9, 30.7, 37.6),
    ('PointBeV',    'gated',   49.1, 68.4,  4.7, 15.6),
    ('CAPE',        'extract', 82.3, 34.3, 57.5, 32.9),
    ('BEVDet',      'extract', 61.5, 16.1, 53.9, 22.8),
    ('BEVDepth',    'extract', 69.5,  5.7, 29.9, 16.6),
    ('LSS',         'extract', 69.1, 24.2, 31.4, 17.1),
    ('GaussianLSS', 'extract', 79.4, 40.6, 37.7, 36.4),
    ('CVT',         'extract', 43.8, 30.4,  3.4,  1.8),
    ('LaRa',        'extract', 65.5, 38.5, 22.5, 21.4),
]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4),
                         gridspec_kw={'width_ratios': [1, 1.25]})

# (a) VP: CAL - EXT
ax = axes[0]
names = [v[0] for v in VP]
gaps = [v[4] - v[3] for v in VP]
cols = [CB if v[2] == 'gated' else CR_ for v in VP]
ys = np.arange(len(VP))[::-1]
ax.barh(ys, gaps, color=cols, height=.62, edgecolor='k', linewidth=.4)
ax.axvline(0, color='k', lw=1)
for y, m, g, v in zip(ys, names, gaps, VP):
    ax.text(-0.6 if g >= 0 else 0.6, y, m, ha='right' if g >= 0 else 'left',
            va='center', fontsize=8.5)
    ax.text(g + (0.45 if g >= 0 else -0.45), y, f'{g:+.1f}',
            ha='left' if g >= 0 else 'right', va='center', fontsize=7.5, color='0.25')
ax.annotate('GaussianLSS: sole exception\n(borderline: all-camera +0.001)',
            xy=(1.6, ys[7]), xytext=(6.0, ys[7] + 1.6), fontsize=7, color='0.35',
            arrowprops=dict(arrowstyle='->', color='0.5', lw=.8))
ax.set_xlabel('mVRS CAL $-$ EXT (points)')
ax.set_title('(a) Viewpoint perturbation', fontsize=10)
ax.set_yticks([])
ax.set_xlim(-14, 16)
ax.text(0.03, 0.97, 'CAL > EXT\n= projection-\nsampling', transform=ax.transAxes,
        fontsize=7.5, va='top', color=CB)
ax.text(0.97, 0.03, 'CAL < EXT\n= extract-\nthen-place', transform=ax.transAxes,
        fontsize=7.5, va='bottom', ha='right', color=CR_)

# (b) CTS: CAL - EXT, suv & bus paired bars
ax = axes[1]
ys = np.arange(len(CTS))[::-1]
for off, idx_e, idx_c, alpha, lab in [(0.18, 2, 3, 1.0, 'SUV'), (-0.18, 4, 5, 0.45, 'Bus')]:
    gaps = [v[idx_c] - v[idx_e] for v in CTS]
    cols = [CB if v[1] == 'gated' else CR_ for v in CTS]
    ax.barh(ys + off, gaps, color=cols, height=.34, alpha=alpha,
            edgecolor='k', linewidth=.4, label=lab)
ax.axvline(0, color='k', lw=1)
for y, v in zip(ys, CTS):
    g = v[3] - v[2]
    ax.text(-1.5 if g >= 0 else 1.5, y, v[0], ha='right' if g >= 0 else 'left',
            va='center', fontsize=8.5)
hs = [plt.Rectangle((0, 0), 1, 1, fc='0.3', alpha=1.0),
      plt.Rectangle((0, 0), 1, 1, fc='0.3', alpha=0.45)]
ax.legend(hs, ['SUV', 'Bus'], fontsize=8, loc='lower right', framealpha=.9)
ax.set_xlabel('CTS CAL $-$ EXT (points)')
ax.set_title('(b) Cross-platform transfer', fontsize=10)
ax.set_yticks([])
ax.set_xlim(-70, 42)
fig.suptitle('The sign of the CAL$-$EXT gap tracks where extrinsics are consumed '
             '— and repeats across both protocols', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(HERE, 'fig_cal_ext_sign.png'), dpi=220)
plt.close(fig)
print('[FigD] sign consistency: VP',
      sum(1 for v in VP if (v[4] > v[3]) == (v[2] == 'gated')), '/', len(VP),
      '| CTS suv', sum(1 for v in CTS if (v[3] > v[2]) == (v[1] == 'gated')), '/', len(CTS),
      '| CTS bus', sum(1 for v in CTS if (v[5] > v[4]) == (v[1] == 'gated')), '/', len(CTS))

# ---------------- Fig Y: magnitude curves (all-camera RRS) ----------------
def co(r): return r.get('condition') or r.get('cond')
def po(r): return r.get('protocol') or r.get('proto')
SRC = {
    'BEVDet':    (f'{R}/BEVDet/vp/eval_vp.json',                 'extract'),
    'BEVDepth':  (f'{R}/BEVDepth/vp/eval_vp.json',               'extract'),
    'CAPE':      (f'{R}/CAPE/vp/vp_cape_sedan_fullframe.json',   'extract'),
    'BEVFormer': (f'{R}/BEVFormer/vp/eval_vp.json',              'gated'),
    'DETR3D':    (f'{R}/DETR3D/vp/eval_vp.json',                 'gated'),
}
def curve(rows, cr, ax_):
    out = {}
    for r in rows:
        if co(r) == cr and r['axis'] == ax_ and po(r) == 'all':
            out[r.get('signed_mag', r.get('mag'))] = r['rrs']
    mags = sorted(out)
    return mags, [out[m] for m in mags]

LS = {'BEVDet': '-', 'BEVDepth': '--', 'CAPE': ':', 'BEVFormer': '-', 'DETR3D': '--'}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
for ax, (cr, axis_, title) in zip(axes, [
        ('CR', 'pitch', '(a) CAL pitch — the mechanism discriminator'),
        ('VR', 'yaw',   '(b) IMG yaw — mechanism-independent knee')]):
    for m, (f, mech) in SRC.items():
        d = json.load(open(f)); rows = d['rows']
        mags, vals = curve(rows, cr, axis_)
        ax.plot(mags, vals, LS[m], color=CB if mech == 'gated' else CR_,
                marker='o', ms=3.2, lw=1.6, label=m)
    ax.axhline(0.5, color='0.75', lw=.8, ls=':')
    ax.text(0, 0.515, 'half', fontsize=7, color='0.5', ha='center')
    ax.set_xlabel('signed magnitude (deg)')
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([-20, -12, -4, 4, 12, 20])
    ax.grid(alpha=.2)
axes[0].set_ylabel('all-camera RRS')
axes[0].set_ylim(-0.03, 1.03)
axes[0].legend(fontsize=7.5, loc='lower center', ncol=2, framealpha=.9)
fig.suptitle('Magnitude response (all-camera): CAL safety exists only for '
             'projection-sampling; IMG collapse is mechanism-independent', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(HERE, 'fig_magnitude_curves.png'), dpi=220)
plt.close(fig)
print('[FigY] written')
