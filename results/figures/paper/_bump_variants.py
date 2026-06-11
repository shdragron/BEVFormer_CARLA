"""Two more-intuitive variants of the EXT->IMG inversion figure (1/7 mVRS)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

EXT = {'CAPE': 94.0, 'BEVDepth': 90.2, 'BEVDet': 89.2, 'DETR3D': 84.2,
       'BEVFormer': 83.1, 'LaRa': 91.1, 'CVT': 90.9, 'GaussianLSS': 90.7,
       'LSS': 88.4, 'PointBeV': 79.4}
IMG = {'CAPE': 83.9, 'DETR3D': 83.9, 'BEVFormer': 83.8, 'BEVDet': 83.0,
       'BEVDepth': 82.8, 'GaussianLSS': 84.5, 'CVT': 83.5, 'LaRa': 82.5,
       'LSS': 81.9, 'PointBeV': 79.6}
DET = ['CAPE', 'BEVDepth', 'BEVDet', 'DETR3D', 'BEVFormer']
SEG = ['LaRa', 'CVT', 'GaussianLSS', 'LSS', 'PointBeV']


def dodge(pairs, min_gap):
    """pairs: (y, key); returns key -> label-y, pushed apart top-down."""
    out = {}
    last = None
    for y, k in sorted(pairs, key=lambda t: -t[0]):
        if last is not None and last - y < min_gap:
            y = last - min_gap
        out[k] = y
        last = y
    return out


# ---------- Variant A: slope chart on the value axis ----------
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
for ax, models, mark, tag in [(axes[0], DET, 'o', 'Detection'),
                              (axes[1], SEG, '^', 'Segmentation')]:
    ax.grid(axis='y', linestyle=':', color='0.88', linewidth=0.7, zorder=0)
    lab_l = dodge([(EXT[m], m) for m in models], 0.85)
    for m in models:
        ax.plot([0, 1], [EXT[m], IMG[m]], '-', color=C(m), lw=1.5, zorder=2,
                solid_capstyle='round')
        ax.plot([0, 1], [EXT[m], IMG[m]], mark, color=C(m), ms=4.5, zorder=3,
                markeredgecolor='k', markeredgewidth=.4)
        yl = lab_l[m]
        if abs(yl - EXT[m]) > 0.05:  # leader line when dodged
            ax.plot([-0.085, -0.02], [yl, EXT[m]], color='0.75', lw=0.5, zorder=1)
        ax.text(-0.10, yl, f'{m}  {EXT[m]:.1f}', ha='right', va='center', fontsize=6.5)
    if tag == 'Detection':
        lo = min(IMG[m] for m in models); hi = max(IMG[m] for m in models)
        ax.plot([1.10, 1.10], [lo, hi], color='0.45', lw=0.9, zorder=2)
        for y in (lo, hi):
            ax.plot([1.075, 1.125], [y, y], color='0.45', lw=0.9, zorder=2)
        ax.text(1.16, (lo + hi) / 2, f'{lo:.1f}–{hi:.1f}', ha='left',
                va='center', fontsize=6.5, color='0.35')
    else:
        lab_r = dodge([(IMG[m], m) for m in models], 0.85)
        for m in models:
            yr = lab_r[m]
            if abs(yr - IMG[m]) > 0.05:
                ax.plot([1.02, 1.085], [IMG[m], yr], color='0.75', lw=0.5, zorder=1)
            ax.text(1.10, yr, f'{IMG[m]:.1f}  {m}', ha='left', va='center', fontsize=6.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['EXT', 'IMG'])
    ax.set_xlim(-0.95, 1.95)
    ax.text(0.0, 1.02, tag, transform=ax.transAxes, fontsize=7, color='0.45',
            ha='left', va='bottom')
axes[0].set_ylabel('mVRS (%)')
axes[0].set_ylim(78.0, 95.8)
fig.tight_layout(w_pad=1.6)
fig.savefig('/tmp/variantA_slope.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- Variant B: EXT - IMG over-report bars ----------
delta = {m: EXT[m] - IMG[m] for m in EXT}
order = sorted(delta, key=lambda m: -delta[m])
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.grid(axis='x', linestyle=':', color='0.88', linewidth=0.7, zorder=0)
ys = range(len(order))[::-1] if False else list(range(len(order), 0, -1))
for y, m in zip(ys, order):
    d = delta[m]
    ax.barh(y, d, height=0.62, color=C(m), zorder=2,
            edgecolor='k', linewidth=0.4)
    mk = 'o' if m in DET else '^'
    xt = d + 0.25 if d >= 0 else d - 0.25
    ax.text(xt, y, f'{d:+.1f}', ha='left' if d >= 0 else 'right',
            va='center', fontsize=6.5)
    ax.text(-0.25 if d >= 0 else 0.25, y, m,
            ha='right' if d >= 0 else 'left', va='center', fontsize=6.5)
ax.axvline(0, color='k', lw=0.8, zorder=3)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('EXT $-$ IMG (mVRS pts)')
ax.set_xlim(-3.2, 12.5)
fig.tight_layout()
fig.savefig('/tmp/variantB_delta.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('done')
