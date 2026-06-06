"""Merge the frame-fixed VR+CR re-run into the committed VP results.

Keeps the still-valid ER rows + Normal from the original committed run; replaces the
(frame-mismatched) VR/CR rows with the re-run's frame-2N-corrected rows. All RRS are
recomputed against a single NDS_Normal for consistency. Writes corrected
eval_vp.json / eval_vp_per_config.csv / eval_vp_summary.txt into results/BEVDepth/vp/.
"""
import csv, json, os, os.path as osp, shutil
import numpy as np

COMMITTED = 'results/BEVDepth/vp/eval_vp.json'
NEW = 'bev_det_benchmark/out/vp_bevdepth_sedan_vrcr/eval_vp.json'
OUTDIR = 'results/BEVDepth/vp'

committed = json.load(open(COMMITTED))
new = json.load(open(NEW))

er_rows = [r for r in committed['rows'] if r['cond'] == 'ER']
vrcr_rows = [r for r in new['rows'] if r['cond'] in ('VR', 'CR')]
assert len(er_rows) == 210, f'ER rows={len(er_rows)} (expect 210)'
assert len(vrcr_rows) == 420, f'VR+CR rows={len(vrcr_rows)} (expect 420)'

nds_norm = new['nds_normal']
m6_norm = new['map6_normal']
print(f'NDS_Normal committed={committed["nds_normal"]:.4f}  re-run={nds_norm:.4f}  '
      f'(diff {abs(committed["nds_normal"]-nds_norm):.4f})')
assert abs(committed['nds_normal'] - nds_norm) < 0.02, 'Normal drifted >0.02 -- investigate'

rows = er_rows + vrcr_rows
for r in rows:                      # single-Normal RRS for consistency
    r['rrs'] = r['nds'] / nds_norm

agg = {}
for cond in ['ER', 'VR', 'CR']:
    percam = [r['rrs'] for r in rows if r['cond'] == cond and r['proto'] != 'all']
    allcam = [r['rrs'] for r in rows if r['cond'] == cond and r['proto'] == 'all']
    mrrs, rrsall = float(np.mean(percam)), float(np.mean(allcam))
    agg[cond] = {'mRRS_percam': mrrs, 'RRSALL_allcam': rrsall,
                 'mVRS': 0.5 * (mrrs + rrsall)}

# preserve the buggy file for provenance
if osp.exists(COMMITTED) and not osp.exists(OUTDIR + '/eval_vp_BUGGY_framemismatch.json'):
    shutil.copy(COMMITTED, OUTDIR + '/eval_vp_BUGGY_framemismatch.json')

js = {'tag': 'bevdepth_sedan', 'frames_per_scene': new['frames_per_scene'],
      'nds_normal': nds_norm, 'map6_normal': m6_norm,
      'note': 'VR/CR re-run with carla_VR frame-2N fix (geobev N == VR 2N); ER+Normal '
              'from the original run (unaffected). RRS recomputed vs this NDS_Normal.',
      'aggregate': agg, 'rows': rows}
json.dump(js, open(osp.join(OUTDIR, 'eval_vp.json'), 'w'), indent=2)

with open(osp.join(OUTDIR, 'eval_vp_per_config.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['condition', 'axis', 'mag', 'protocol', 'nds', 'map6', 'rrs'])
    for r in rows:
        w.writerow([r['cond'], r['axis'], r['mag'], r['proto'],
                    f"{r['nds']:.4f}", f"{r['map6']:.4f}", f"{r['rrs']:.4f}"])

lines = ['VP viewpoint-robustness (BEVDepth NDS)   tag=bevdepth_sedan  [carla_VR frame-2N FIX]',
         f'  frames-per-scene={new["frames_per_scene"]}  NDS_Normal={nds_norm:.4f}  mAP6_Normal={m6_norm:.4f}',
         '  RRS = NDS_cell / NDS_Normal   [VR primary]   (ER+Normal from orig run; VR/CR frame-fixed)', '',
         '  condition   mRRS(per-cam)   RRSALL(all-cam)        mVRS']
for cond in ['ER', 'VR', 'CR']:
    a = agg[cond]; star = '  <- primary' if cond == 'VR' else ''
    lines.append(f'  {cond:<10}{a["mRRS_percam"]:>13.4f}{a["RRSALL_allcam"]:>16.4f}{a["mVRS"]:>12.4f}{star}')
txt = '\n'.join(lines)
open(osp.join(OUTDIR, 'eval_vp_summary.txt'), 'w').write(txt + '\n')
print('\n' + txt)
print('\n=== buggy (committed) aggregate for comparison ===')
for cond in ['ER', 'VR', 'CR']:
    a = committed['aggregate'][cond]
    print(f'  {cond:<10} mRRS={a["mRRS_percam"]:.4f}  RRSALL={a["RRSALL_allcam"]:.4f}  mVRS={a["mVRS"]:.4f}')
