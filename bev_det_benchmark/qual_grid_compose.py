"""RoboGeo vertical qual figure: conditions (a)-(d) stacked; each condition is a
2x2 of [GT | BEVDepth ; DFA3D | CAPE] six-camera panels, GT(green)+pred(red).
Each panel framed in its MODEL color; a top legend explains color -> model.
Only bold (a)-(d) labels otherwise (no per-panel text).
"""
import os, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = '/home/hanyan_arch/viewpoint/BEVFormer/results/qual_grid'
CONDS = ['a', 'b', 'c', 'd']
ROWS = ['pitch12_img', 'yaw8_img', 'suv_cal', 'bus_cal']     # a,b,c,d order
PANELS = ['GT', 'bevdepth', 'dfa3d', 'cape']                 # 2x2: GT,BEVDepth / DFA3D,CAPE
PW, PH = 960, 360          # six-cam panel size (8:3)
LBL = 150                  # left margin for (a)-(d)
GAP = 22                   # gap between condition blocks
BD2 = 14                   # per-panel colored border thickness
LEG_H = 96                 # top legend band height

# per-model border color (BGR) + legend label
PANEL_COLOR = {'GT': (100, 100, 100), 'bevdepth': (0, 140, 255),
               'dfa3d': (255, 120, 30), 'cape': (200, 60, 170)}
PANEL_NAME = {'GT': 'GT (Ground Truth)', 'bevdepth': 'BEVDepth',
              'dfa3d': 'DFA3D', 'cape': 'CAPE'}


def _font(sz):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf']:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    try:
        import matplotlib
        return ImageFont.truetype(matplotlib.font_manager.findfont('DejaVu Sans:bold'), sz)
    except Exception:
        return ImageFont.load_default()


def load6(model, row):
    p = f'{OUT}/{model}_{row}_6cam.png'
    img = cv2.imread(p)
    if img is None:
        img = np.full((1800, 4800, 3), 35, np.uint8)
        cv2.putText(img, f'MISSING {model}/{row}', (120, 920), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
    return cv2.resize(img, (PW, PH))


def block(row):
    """2x2 six-cam panels, each framed in its own model color. -> (2*PH x 2*PW)."""
    canvas = np.full((2 * PH, 2 * PW, 3), 255, np.uint8)
    for k, model in enumerate(PANELS):
        r, c = divmod(k, 2)
        y0, x0 = r * PH, c * PW
        canvas[y0:y0 + PH, x0:x0 + PW] = load6(model, row)
        col = PANEL_COLOR[model]
        h = BD2 // 2
        cv2.rectangle(canvas, (x0 + h, y0 + h), (x0 + PW - 1 - h, y0 + PH - 1 - h), col, BD2)
    return canvas


def draw_legend(canvas):
    """Top band: swatch + model name for each panel color."""
    sw = 46                                   # swatch size
    pad, txtgap, gap = 18, 14, 60             # paddings
    fnt = _font(34)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    # measure total width
    items = [(m, PANEL_NAME[m]) for m in PANELS]
    widths = []
    for _, name in items:
        tb = dr.textbbox((0, 0), name, font=fnt)
        widths.append(sw + txtgap + (tb[2] - tb[0]))
    total = sum(widths) + gap * (len(items) - 1)
    x = LBL + (2 * PW - total) // 2
    ycen = LEG_H // 2
    for (model, name), w in zip(items, widths):
        b, g, r = PANEL_COLOR[model]
        dr.rectangle([x, ycen - sw // 2, x + sw, ycen + sw // 2], fill=(r, g, b), outline=(0, 0, 0), width=2)
        tb = dr.textbbox((0, 0), name, font=fnt)
        dr.text((x + sw + txtgap, ycen - (tb[3] - tb[1]) // 2 - tb[1]), name, font=fnt, fill=(0, 0, 0))
        x += w + gap
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    blocks = [block(row) for row in ROWS]
    bh, bw = blocks[0].shape[:2]
    H = LEG_H + len(blocks) * bh + GAP * (len(blocks) - 1)
    W = LBL + bw
    canvas = np.full((H, W, 3), 255, np.uint8)
    ys = []
    y = LEG_H
    for b in blocks:
        canvas[y:y + bh, LBL:LBL + bw] = b
        ys.append(y); y += bh + GAP
    canvas = draw_legend(canvas)
    # bold (a)-(d) labels via PIL
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil); fnt = _font(64)
    for letter, y0 in zip(CONDS, ys):
        dr.text((18, y0 + bh // 2 - 40), f'({letter})', font=fnt, fill=(0, 0, 0))
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    p = f'{OUT}/ROBOGEO_QUAL_6CAM.png'
    cv2.imwrite(p, canvas)
    print(f'wrote {p}  ({W}x{H})')


if __name__ == '__main__':
    main()
