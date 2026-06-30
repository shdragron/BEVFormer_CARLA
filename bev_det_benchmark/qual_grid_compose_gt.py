"""RoboGeo dataset-example figure: conditions (a)-(d) stacked vertically, each a
six-camera (2x3) grid with GT(green, visibility>=2) overlay. No models/predictions.
Reads results/qual_grid/GT_<row>_6cam.png.
"""
import os, cv2, numpy as np

OUT = '/home/hanyan_arch/viewpoint/BEVFormer/results/qual_grid'
CONDS = [('a', 'pitch12_img', 'IMG  pitch +12  (front cam)'),
         ('b', 'yaw8_img',    'IMG  yaw +8  (all cam)'),
         ('c', 'suv_cal',     'SUV - CAL'),
         ('d', 'bus_cal',     'BUS - CAL')]
PW, PH = 1860, 698        # six-cam panel display (8:3)
LBL = 120                 # left margin for (a)-(d)
GAP = 14


def load6(row):
    p = f'{OUT}/GT_{row}_6cam.png'
    img = cv2.imread(p)
    if img is None:
        img = np.full((1800, 4800, 3), 35, np.uint8)
        cv2.putText(img, f'MISSING GT/{row}', (120, 920), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
    return cv2.resize(img, (PW, PH))


def main():
    H = len(CONDS) * PH + GAP * (len(CONDS) - 1)
    W = LBL + PW
    canvas = np.full((H, W, 3), 255, np.uint8)
    y = 0
    for letter, row, name in CONDS:
        canvas[y:y + PH, LBL:LBL + PW] = load6(row)
        cv2.rectangle(canvas, (LBL, y), (LBL + PW - 1, y + PH - 1), (0, 0, 0), 1)
        cv2.putText(canvas, f'({letter})', (14, y + PH // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        for j, t in enumerate(name.split('  ')):
            cv2.putText(canvas, t, (10, y + PH // 2 + 30 + j * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        y += PH + GAP
    p = f'{OUT}/ROBOGEO_QUAL_GT_6CAM.png'
    cv2.imwrite(p, canvas)
    print(f'wrote {p}  ({W}x{H})')


if __name__ == '__main__':
    main()
