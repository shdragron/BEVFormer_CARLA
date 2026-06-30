"""Per-model mVRS(%) vs perturbation angle theta, for 3D object detection.

Matches the BEV-Segmentation companion figure exactly (one stacked panel per
model; three lines Ext/Img/Cal; gray-x / blue-o / green-o; shared x-axis).

y(theta) = mean over all 7 protocols (6 single-camera + all-camera) x 3 axes
(roll/pitch/yaw) of RRS at signed magnitude theta, x100. theta=0 -> 100 (clean).
This is the same aggregation as the headline mVRS (mean over theta reproduces the
paper Table-2 values, e.g. BEVFormer EXT 83.2 / IMG 83.9 / CAL 93.6).

Sources = full-3792 per-config CSVs (DFA3D rows[]; BEVFormer per-cam + allcam).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
CONDMAP = {"ER": "EXT", "VR": "IMG", "CR": "CAL"}
MAGS = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20]

C_EXT, C_IMG, C_CAL = '0.5', '#1f77b4', '#2ca02c'


def _rows_csv(p, magcol):
    out = []
    for r in csv.DictReader(open(p)):
        c = CONDMAP.get(r["condition"].strip())
        if not c:
            continue
        out.append((c, int(float(r[magcol])), r["protocol"].strip(), float(r["rrs"])))
    return out


def _rows_json(p):
    out = []
    for r in json.load(open(p))["rows"]:
        c = CONDMAP.get(r["condition"])
        if not c:
            continue
        out.append((c, int(r["signed_mag"]), r["protocol"], float(r["rrs"])))
    return out


SRC = {
    "BEVDet":   _rows_csv(f"{R}/BEVDet/vp/eval_vp_per_config.csv", "mag"),
    "BEVDepth": _rows_csv(f"{R}/BEVDepth/vp/eval_vp_per_config.csv", "mag"),
    "BEVFormer": _rows_csv(f"{R}/BEVFormer/vp/eval_vp_full_percam_per_config.csv", "signed_mag")
               + _rows_csv(f"{R}/BEVFormer/vp/eval_vp_full3792_allcam_per_config.csv", "signed_mag"),
    "DFA3D":    _rows_json(f"{R}/DFA3D/vp/vp_dfa3d_sedan_fullframe.json"),
    "DETR3D":   _rows_csv(f"{R}/../bev_det_benchmark/sparse/out_vp_detr3d/vp_detr3d_sedan_full/eval_vp_per_config.csv", "mag"),
    "CAPE":     _rows_csv(f"{R}/CAPE/vp/vp_cape_sedan_fullframe_per_config.csv", "mag"),
}
# row-order in stacked figure (forward | backward | projection-free)
MODELS = ["BEVDet", "BEVDepth", "BEVFormer", "DFA3D", "DETR3D", "CAPE"]


def curve(rows, cond):
    ys = []
    for mg in MAGS:
        if mg == 0:
            ys.append(100.0)
            continue
        v = [r[3] for r in rows if r[0] == cond and r[1] == mg]
        ys.append(sum(v) / len(v) * 100)
    return ys


plt.rcParams.update({
    'font.size': 8, 'axes.linewidth': 0.7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 7,
})

n = len(MODELS)
fig, axes = plt.subplots(n, 1, figsize=(3.6, 1.25 * n), sharex=True)
# only inter-panel spacing here; bbox_inches='tight' at save time expands the
# canvas to include the left model labels and top legend (nothing clipped)
fig.subplots_adjust(left=0.24, right=0.97, top=0.93, bottom=0.07, hspace=0.38)

for ax, m in zip(axes, MODELS):
    rows = SRC[m]
    ax.plot(MAGS, curve(rows, "EXT"), color=C_EXT, marker='x', ms=3.5,
            lw=1.2, mew=1.0, zorder=3)
    ax.plot(MAGS, curve(rows, "IMG"), color=C_IMG, marker='o', ms=3.0,
            lw=1.2, zorder=3)
    ax.plot(MAGS, curve(rows, "CAL"), color=C_CAL, marker='o', ms=3.0,
            lw=1.2, zorder=3)
    ax.set_xticks(MAGS)
    ax.set_yticks([80, 90, 100])
    ax.set_ylim(73, 102)
    ax.set_xlim(-21.5, 21.5)
    ax.margins(x=0)
    ax.grid(color='0.88', linewidth=0.6, zorder=0)
    ax.tick_params(length=2)
    ax.text(-0.215, 0.5, m, transform=ax.transAxes, ha='right', va='center',
            fontsize=9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

axes[-1].set_xlabel(r'Perturbation angle $\theta$ (deg)', fontsize=8.5)

handles = [
    plt.Line2D([], [], color=C_EXT, marker='x', mew=1.0, ms=5, lw=1.2, label='Ext'),
    plt.Line2D([], [], color=C_IMG, marker='o', ms=4, lw=1.2, label='Img'),
    plt.Line2D([], [], color=C_CAL, marker='o', ms=4, lw=1.2, label='Cal'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.6, 0.935), handletextpad=0.4, columnspacing=1.6,
           fontsize=8.5)

for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'fig_vp_angle_curves_det.{ext}'),
                dpi=300 if ext == 'png' else None,
                bbox_inches='tight', pad_inches=0.12)
plt.close(fig)
print('wrote fig_vp_angle_curves_det.{pdf,png}')
