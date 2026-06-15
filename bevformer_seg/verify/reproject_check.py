"""Reproject GT vehicle BEV cells into the 6 camera images to validate the full
geometry chain (BEV grid -> ego 3D -> image via lidar2img = K4 @ E), plus a BEV
panel showing the GT in the exact grid the model is trained on (g[::-1,::-1].T).

Picks the frame whose vis>=2 vehicle cells are spread across the most cameras,
so left/right/front mirroring would be obvious.

BEV->ego (GaussianLSS/CVT view = [[0,-2,100],[-2,0,100],[0,0,1]], 0.5 m/cell):
    col = 100 - 2*y  ->  y = (100 - col)/2 ;  row = 100 - 2*x  ->  x = (100 - row)/2
"""
import warnings; warnings.filterwarnings('ignore')
import glob, json, os
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

GAUSS = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev_labels/gaussianlss'
IMG_ROOT = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev'
LABELS = f'{GAUSS}/sedan_eval'
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
COL = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
SRC_W, SRC_H, RS_W, RS_H, TOP_CROP = 1600.0, 900.0, 480.0, 270.0, 46
VEH_BIT, MIN_VIS = 4, 2


def rescale_K(I):
    I = np.array(I, dtype=np.float64).copy()
    I[0, 0] *= RS_W / SRC_W; I[0, 2] *= RS_W / SRC_W
    I[1, 1] *= RS_H / SRC_H; I[1, 2] *= RS_H / SRC_H
    I[1, 2] -= TOP_CROP
    return I


def project(x, y, P, z=0.0):
    """Project ego ground points (z meters, ground = 0) to model-image pixels."""
    pts = np.stack([x, y, np.full_like(x, z), np.ones_like(x)], 0)
    h = P @ pts
    zc = h[2]; m = zc > 0.1
    uu = h[0][m] / zc[m]; vv = h[1][m] / zc[m]
    ib = (uu >= 0) & (uu < RS_W) & (vv >= 0) & (vv < RS_H - TOP_CROP)
    return uu[ib], vv[ib]


scenes = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(LABELS, 'scene_*.json')))


def veh_vis_xy(scene, fr):
    bev = np.array(Image.open(os.path.join(LABELS, scene, fr['bev'])))
    vis = np.array(Image.open(os.path.join(LABELS, scene, fr['visibility'])))
    m = (((bev >> VEH_BIT) & 1).astype(bool)) & (vis >= MIN_VIS)
    rows, cols = np.where(m)
    return (100.0 - (rows + 0.5)) / 2.0, (100.0 - (cols + 0.5)) / 2.0, m


# scan for the frame whose cells hit the most distinct cameras
best = None
for scene in scenes[:30]:
    for fr in json.load(open(os.path.join(LABELS, f'{scene}.json'))):
        x, y, m = veh_vis_xy(scene, fr)
        if len(x) < 20:
            continue
        Ps = []
        hit = 0
        for ci in range(6):
            K4 = np.eye(4); K4[:3, :3] = rescale_K(fr['intrinsics'][ci])
            P = K4 @ np.array(fr['extrinsics'][ci], dtype=np.float64)
            u, v = project(x, y, P)
            Ps.append(P)
            if len(u) > 0:
                hit += 1
        score = (hit, len(x))
        if best is None or score > best[0]:
            best = (score, scene, fr, x, y, m, Ps)

(_, scene, fr, x, y, m, Ps) = best
print(f'chosen {scene} {fr["token"][:8]}  cams_hit={best[0][0]}  cells={best[0][1]}')

fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 4)
# BEV panel (model grid): g[::-1,::-1].T, ego at center, front = +x = up
bev_grid = m[::-1, ::-1].T
axb = fig.add_subplot(gs[:, 0])
axb.imshow(bev_grid, cmap='gray', origin='lower')
axb.plot(100, 100, 'r+', ms=12)
axb.annotate('front', (100, 140), color='yellow', ha='center', fontsize=8)
axb.set_title('GT in model grid\n(g[::-1,::-1].T)', fontsize=9); axb.axis('off')

cam_axes = [gs[0, 1], gs[0, 2], gs[0, 3], gs[1, 1], gs[1, 2], gs[1, 3]]
for gci, ci in zip(cam_axes, range(6)):
    ax = fig.add_subplot(gci)
    fn = os.path.join(IMG_ROOT, fr['images'][ci])
    im = Image.open(fn).convert('RGB').resize((int(RS_W), int(RS_H)), Image.BILINEAR)
    im = im.crop((0, TOP_CROP, int(RS_W), int(RS_H)))
    ax.imshow(np.asarray(im)); ax.set_title(CAMS[ci], fontsize=9); ax.axis('off')
    ug, vg = project(x, y, Ps[ci], z=0.0)       # ground footprint (tire line)
    ur, vr = project(x, y, Ps[ci], z=1.6)       # roof
    if len(ur):
        ax.scatter(ur, vr, s=2, c=COL[ci], alpha=0.18)
    if len(ug):
        ax.scatter(ug, vg, s=3, c=COL[ci], alpha=0.7)
fig.suptitle(f'{scene} {fr["token"][:8]} — vis>=2 vehicle cells reprojected '
             f'[{int(m.sum())} cells]', fontsize=11)
fig.tight_layout()
out = 'bevformer_seg/out/reproject_check.png'
fig.savefig(out, dpi=95)
print('wrote', out)
