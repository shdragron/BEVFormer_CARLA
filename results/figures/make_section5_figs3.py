"""New Fig. C for the v4 §5.3: which VP component predicts cross-platform transfer.

fig_vp_cts_alignment.png
 (a) all-camera IMG RRS (x) vs SUV CTS-IMG % (y): tight alignment (det rho=1.00, seg 0.90)
 (b) Spearman rho vs SUV CTS for three VP predictors:
     all-camera IMG / per-camera IMG / all-camera EXT  (det & seg bars)
     -> IMG aligns, per-camera dilutes, EXT weakens (seg) or inverts (det).

det VP all-cam values: full-3792 for BEVDet/BEVDepth/CAPE, subset768 for BEVFormer/DETR3D
(verified 2026-06-10). CTS = corrected ratios (BEVDet oracle-normalized). seg per-camera
and EXT correlations computed from eval_results jsons. SimpleBEV excluded.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, glob, os
import statistics as st
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
CB, CR_ = '#1f77b4', '#d62728'

# ---------------- data ----------------
DET = ['BEVFormer', 'DETR3D', 'CAPE', 'BEVDet', 'BEVDepth']
MECH_D = {'BEVFormer': 'gated', 'DETR3D': 'gated', 'CAPE': 'extract',
          'BEVDet': 'extract', 'BEVDepth': 'extract'}
det_img = {'BEVFormer': .426, 'DETR3D': .422, 'CAPE': .400, 'BEVDet': .360, 'BEVDepth': .325}
det_ext = {'BEVFormer': .428, 'DETR3D': .438, 'CAPE': .803, 'BEVDet': .610, 'BEVDepth': .648}
det_pc  = {'BEVFormer': .907, 'DETR3D': .909, 'CAPE': .912, 'BEVDet': .908, 'BEVDepth': .912}
det_cts = {'BEVFormer': 37.2, 'DETR3D': 34.9, 'CAPE': 33.8, 'BEVDet': 17.0, 'BEVDepth': 10.3}

SEG = ['GaussianLSS', 'CVT', 'LaRa', 'LSS', 'PointBeV']
MECH_S = {'GaussianLSS': 'extract', 'CVT': 'extract', 'LaRa': 'extract',
          'LSS': 'extract', 'PointBeV': 'gated'}
KEY = {'GaussianLSS': 'glss', 'CVT': 'cvt', 'LaRa': 'lara', 'LSS': 'lss', 'PointBeV': 'pointbev'}
seg_cts = {'CVT': 44.3, 'GaussianLSS': 36.1, 'LaRa': 31.0, 'LSS': 25.0, 'PointBeV': 20.2}
seg_img, seg_ext, seg_pc = {}, {}, {}
for m in SEG:
    f = sorted(glob.glob(f'{R}/eval_results/bevunify-{KEY[m]}-carla/vr_*/eval_vr.json'))[-1]
    d = json.load(open(f)); mv = d['mVRS']
    seg_img[m] = mv['RRSALL_IMG_allCam']
    seg_ext[m] = mv['RRSALL_EXT_allCam']
    seg_pc[m]  = mv['mRRS_IMG_perCam']

def rho(xd, yd, models):
    return sps.spearmanr([xd[m] for m in models], [yd[m] for m in models]).statistic

rhos = {
    'det': [rho(det_img, det_cts, DET), rho(det_pc, det_cts, DET), rho(det_ext, det_cts, DET)],
    'seg': [rho(seg_img, seg_cts, SEG), rho(seg_pc, seg_cts, SEG), rho(seg_ext, seg_cts, SEG)],
}
print('[FigC2] det rho (allcam-IMG, percam-IMG, allcam-EXT) vs SUV CTS:',
      [f'{v:+.2f}' for v in rhos['det']])
print('[FigC2] seg rho:', [f'{v:+.2f}' for v in rhos['seg']])

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), gridspec_kw={'width_ratios': [1.15, 1]})

# (a) scatter: all-cam IMG vs SUV CTS
ax = axes[0]
OFF = {'BEVFormer': (.006, 0.6), 'DETR3D': (.006, -1.6), 'CAPE': (.006, .6),
       'BEVDet': (.006, .4), 'BEVDepth': (.006, .4),
       'GaussianLSS': (.006, .5), 'CVT': (.006, .5), 'LaRa': (.006, .5),
       'LSS': (.006, .5), 'PointBeV': (.006, .5)}
for m in DET:
    ax.scatter(det_img[m], det_cts[m], c=CB if MECH_D[m] == 'gated' else CR_,
               marker='o', s=64, edgecolors='k', linewidths=.5, zorder=3)
    ax.annotate(m, (det_img[m] + OFF[m][0], det_cts[m] + OFF[m][1]), fontsize=7.5)
for m in SEG:
    ax.scatter(seg_img[m], seg_cts[m], c=CB if MECH_S[m] == 'gated' else CR_,
               marker='^', s=64, edgecolors='k', linewidths=.5, zorder=3)
    ax.annotate(m, (seg_img[m] + OFF[m][0], seg_cts[m] + OFF[m][1]), fontsize=7.5)
ax.set_xlabel('VP all-camera IMG (RRS)')
ax.set_ylabel('SUV CTS$_{\\mathrm{IMG}}$ (%)')
ax.set_title('(a) All-camera IMG aligns with SUV transfer', fontsize=9.5)
ax.text(0.03, 0.97,
        f'rank corr.  det $\\rho$={rhos["det"][0]:+.2f}\n'
        f'               seg $\\rho$={rhos["seg"][0]:+.2f}',
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(fc='white', ec='0.8', alpha=.9))
hs = [plt.Line2D([], [], color=CB, marker='s', ls='', label='projection-sampling'),
      plt.Line2D([], [], color=CR_, marker='s', ls='', label='extract-then-place'),
      plt.Line2D([], [], color='k', marker='o', ls='', label='detection'),
      plt.Line2D([], [], color='k', marker='^', ls='', label='segmentation')]
ax.legend(handles=hs, fontsize=7, loc='lower right', framealpha=.9)
ax.grid(alpha=.2)

# (b) bar: rho per predictor
ax = axes[1]
labels = ['all-camera\nIMG', 'per-camera\nIMG', 'all-camera\nEXT']
x = np.arange(3)
w = 0.36
ax.bar(x - w/2, rhos['det'], w, color='0.25', label='detection', edgecolor='k', linewidth=.4)
ax.bar(x + w/2, rhos['seg'], w, color='0.65', label='segmentation', edgecolor='k', linewidth=.4)
for xi, v in zip(x - w/2, rhos['det']):
    ax.text(xi, v + (.05 if v >= 0 else -.10), f'{v:+.2f}', ha='center', fontsize=8)
for xi, v in zip(x + w/2, rhos['seg']):
    ax.text(xi, v + (.05 if v >= 0 else -.10), f'{v:+.2f}', ha='center', fontsize=8)
ax.axhline(0, color='k', lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Spearman $\\rho$ vs SUV CTS$_{\\mathrm{IMG}}$')
ax.set_ylim(-1.05, 1.2)
ax.set_title('(b) Which VP component predicts transfer', fontsize=9.5)
ax.legend(fontsize=8, loc='lower left', framealpha=.9)
ax.grid(axis='y', alpha=.2)

fig.suptitle('Re-rendered all-camera IMG is the closest in-platform proxy for '
             'cross-platform transfer; extrinsic-only evaluation inverts it', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(HERE, 'fig_vp_cts_alignment.png'), dpi=220)
plt.close(fig)
print('[FigC2] written fig_vp_cts_alignment.png')
