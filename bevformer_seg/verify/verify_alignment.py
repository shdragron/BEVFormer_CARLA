"""Lock the GaussianLSS-GT <-> BEVFormer-grid alignment before any training.

Loads ONE frame, builds lidar2img the GaussianLSS way (viewpad(I_rescaled) @ E,
E = ego2cam), then:
  (a) renders the vehicle GT mask (bit0 of bev png) and its flip to the
      BEVFormer grid layout (row=lidarY+, col=lidarX+);
  (b) projects a synthetic 3D point 12 m straight ahead (ego +X) into all 6
      cameras -> must land near the centre of CAM_FRONT and nowhere else;
  (c) for each GT vehicle pixel, back-maps to ego metres and forward-projects
      to cameras -> overlays should fall on actual vehicles in the images.

Output: out/align_<scene>_<frame>.png (6 cam images + GT-raw + GT-bevformer).
"""
import json, os, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LBL = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev_labels/gaussianlss/sedan'
IMG_ROOT = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev'
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
TOP_CROP = 46
IMG_H, IMG_W = 224, 480
SRC_W, SRC_H = 1600.0, 900.0


def rescale_intrinsic(I):
    """GaussianLSS transforms.py:162-167: resize 1600x900 -> 480x270, crop 46."""
    w_resize, h_resize = 480.0, 270.0
    I = np.array(I, dtype=np.float64).copy()
    I[0, 0] *= w_resize / SRC_W
    I[0, 2] *= w_resize / SRC_W
    I[1, 1] *= h_resize / SRC_H
    I[1, 2] *= h_resize / SRC_H
    I[1, 2] -= TOP_CROP
    return I


def load_image(rel):
    im = Image.open(os.path.join(IMG_ROOT, rel)).convert('RGB')
    im = im.resize((480, 270), Image.BILINEAR).crop((0, TOP_CROP, 480, 270))
    return np.asarray(im)                       # (224,480,3)


def vehicle_mask(scene, bev_png):
    arr = np.array(Image.open(os.path.join(LBL, scene, bev_png)))  # uint16 200x200
    return ((arr >> 4) & 1).astype(np.uint8)            # bit4 = vehicle (verified)


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else 'scene_0006'
    fi = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    frames = json.load(open(os.path.join(LBL, f'{scene}.json')))
    d = frames[fi]
    print(f'[verify] {scene} frame {fi} token {d["token"][:8]}')

    # lidar2img per cam (GaussianLSS-exact)
    l2i = []
    for i in range(6):
        I = rescale_intrinsic(d['intrinsics'][i])
        E = np.array(d['extrinsics'][i], dtype=np.float64)   # ego2cam (4x4)
        K4 = np.eye(4); K4[:3, :3] = I
        l2i.append(K4 @ E)
    l2i = np.stack(l2i)                                       # (6,4,4)

    # (b) synthetic point 12 m straight ahead, 0.5 m above ground (ego +X fwd)
    P = np.array([12.0, 0.0, -0.5, 1.0])                      # ego/lidar frame
    print('[verify] project ego point (12,0,-0.5) [forward]:')
    for i, cam in enumerate(CAMS):
        uv = l2i[i] @ P
        if uv[2] > 0.1:
            u, v = uv[0] / uv[2], uv[1] / uv[2]
            inb = 0 <= u <= IMG_W and 0 <= v <= IMG_H
            print(f'   {cam:16s} u={u:7.1f} v={v:6.1f} {"<-- IN FRAME" if inb else ""}')

    # (a) GT mask: raw and BEVFormer-aligned (verified: bev[r,c]=g[199-c,199-r])
    g = vehicle_mask(scene, d['bev'])
    bevf = g[::-1, ::-1].T                        # np.flip(g).T == g[199-c,199-r]
    print(f'[verify] GT vehicle pixels: {g.sum()}')

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, cam in enumerate(CAMS):
        ax = axes[i // 3 if i < 3 else 1, i % 3]  # rough placement
    # simpler: 6 cams in a 2x3 block, 2 GT panels on the right column
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    order = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    for i, cam in enumerate(CAMS):
        r, c = order[i]
        img = load_image(d['images'][i])
        axes[r, c].imshow(img)
        # overlay projected GT-vehicle pixels (subsample)
        ys, xs = np.where(g > 0)
        sub = np.linspace(0, len(xs)-1, min(400, len(xs))).astype(int) if len(xs) else []
        for k in sub:
            # GT pixel (row=ys,col=xs) -> ego metres via inverse view matrix
            # view: col=-2Y+100, row=-2X+100 -> X=(100-row)/2, Y=(100-col)/2
            X = (100 - ys[k]) / 2.0; Y = (100 - xs[k]) / 2.0
            uv = l2i[i] @ np.array([X, Y, -0.5, 1.0])
            if uv[2] > 0.1:
                u, v = uv[0]/uv[2], uv[1]/uv[2]
                if 0 <= u < IMG_W and 0 <= v < IMG_H:
                    axes[r, c].plot(u, v, '.', ms=2, color='lime')
        axes[r, c].set_title(cam, fontsize=9); axes[r, c].axis('off')
    axes[0, 3].imshow(g, cmap='gray'); axes[0, 3].set_title('GT raw (GaussianLSS layout)')
    axes[1, 3].imshow(bevf, cmap='gray'); axes[1, 3].set_title('GT -> BEVFormer grid (Y+ row, X+ col)')
    # mark forward (ego +X) direction on the BEVFormer-grid panel
    axes[1, 3].plot(100, 100, 'r+', ms=12); axes[1, 3].annotate('ego', (100, 100), color='r')
    for a in (axes[0,3], axes[1,3]): a.axis('on')
    fig.tight_layout()
    out = f'/home/hanyan_arch/viewpoint/BEVFormer/bevformer_seg/out/align_{scene}_{fi}.png'
    fig.savefig(out, dpi=90); print('[verify] wrote', out)


if __name__ == '__main__':
    main()
