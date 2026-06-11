"""Recompose focused per-axis / per-target condition strips from the existing
GT+pred 6-view grids (no re-inference). Each strip = CAM_FRONT of
[NORMAL | EXT | IMG | CAL] side by side, so the pred change across conditions is
directly comparable on one camera (NORMAL = reference; EXT on the clean image,
IMG/CAL on the tilted image).
"""
import os, sys, cv2, numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/hanyan_arch/viewpoint/BEVFormer/results/_qual_conditions/scene-0267-frame-0016_pred_thr0.3'
MAG = 20
AXES = ['roll', 'pitch', 'yaw']
COND_LABEL = {'NORMAL': 'NORMAL (ref)', 'EXT': 'EXT  (img=normal)',
              'IMG': 'IMG  (img=tilted)', 'CAL': 'CAL  (img=tilted)'}
PANEL_W = 1000


def front(grid_name):
    g = cv2.imread(os.path.join(D, grid_name))
    if g is None:
        return None
    return g[0:900, 1600:3200]                       # CAM_FRONT cell of the 2x3 grid


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(img, text, (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
    return img


def strip(panels, labels, outname):
    cells = []
    for p, lb in zip(panels, labels):
        if p is None:
            p = np.zeros((900, 1600, 3), np.uint8)
        c = label(p, lb)
        h = int(c.shape[0] * PANEL_W / c.shape[1])
        cells.append(cv2.resize(c, (PANEL_W, h)))
    out = np.hstack(cells)
    cv2.imwrite(os.path.join(D, outname), out, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f'  {outname}  {out.shape[1]}x{out.shape[0]}')
    return out


def main():
    # VP: one strip per axis
    for ax in AXES:
        panels = [front('VP_NORMAL_6view.jpg'),
                  front(f'VP_EXT_{ax}+{MAG}_6view.jpg'),
                  front(f'VP_IMG_{ax}+{MAG}_6view.jpg'),
                  front(f'VP_CAL_{ax}+{MAG}_6view.jpg')]
        labels = [COND_LABEL['NORMAL'], f'EXT {ax}+{MAG} (img=normal)',
                  f'IMG {ax}+{MAG} (img=tilted)', f'CAL {ax}+{MAG} (img=tilted)']
        strip(panels, labels, f'STRIP_VP_{ax}+{MAG}.jpg')
    # CTS: one strip per target
    for tgt in ['suv', 'bus']:
        panels = [front(f'CTS-{tgt}_NORMAL_6view.jpg'),
                  front(f'CTS-{tgt}_EXT_6view.jpg'),
                  front(f'CTS-{tgt}_IMG_6view.jpg'),
                  front(f'CTS-{tgt}_CAL_6view.jpg')]
        labels = [f'CTS-{tgt} NORMAL (ref)', f'CTS-{tgt} EXT (img=sedan)',
                  f'CTS-{tgt} IMG (img={tgt})', f'CTS-{tgt} CAL (img={tgt})']
        strip(panels, labels, f'STRIP_CTS-{tgt}.jpg')
    print('DONE')


if __name__ == '__main__':
    main()
