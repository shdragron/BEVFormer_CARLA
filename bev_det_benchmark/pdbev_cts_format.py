"""Format the PD-BEV CTS results (out/cts_pdbev_sedan384/cts_nds.tsv + the measured
native-384 oracles) into the standard results/<Model>/cts/ files
(eval_cts.csv, eval_cts.json, eval_cts_summary.txt), matching the other detectors.

CTS_c = NDS(c) / P_TARGET (paper Eq.6). P_TARGET = the native-384 target oracle
(suv 0.5464, bus 0.6064). IMG is the primary cross-platform condition.

Note: per-condition TP-error components (mATE/...) were not retained from the CTS
run (dets pkls freed after scoring); only NDS/mAP6/CTS are emitted. Re-run
pdbev_cts_run.sh capturing the full [CARLA-METRICS-JSON] if components are needed.
"""
import csv
import json
import os
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
SRC = osp.join(HERE, 'out', 'cts_pdbev_sedan384', 'cts_nds.tsv')
OUT = '/home/hanyan_arch/viewpoint/BEVFormer/results/PDBEV/cts'
ORACLE = {'suv': 0.5464, 'bus': 0.6064}
CONDS = ['NORMAL', 'EXT', 'IMG', 'CAL']


def main():
    nds, m6 = {}, {}
    for r in csv.DictReader(open(SRC), delimiter='\t'):
        nds[(r['target'], r['cond'])] = float(r['nds'])
        m6[(r['target'], r['cond'])] = float(r['map'])
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for T in ('suv', 'bus'):
        P = ORACLE[T]
        rows.append({'platform': T, 'condition': 'ORACLE', 'nds': P,
                     'map6': None, 'cts_nds': 1.0, 'primary': ''})
        for C in CONDS:
            v = nds.get((T, C))
            rows.append({'platform': T, 'condition': C, 'nds': v,
                         'map6': m6.get((T, C)),
                         'cts_nds': (v / P) if v is not None else None,
                         'primary': 'IMG' if C == 'IMG' else ''})

    with open(osp.join(OUT, 'eval_cts.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['platform', 'condition', 'nds', 'map6', 'cts_nds', 'primary'])
        for r in rows:
            w.writerow([r['platform'], r['condition'],
                        f"{r['nds']:.4f}" if r['nds'] is not None else '',
                        f"{r['map6']:.4f}" if r['map6'] is not None else '',
                        f"{r['cts_nds']:.4f}" if r['cts_nds'] is not None else '',
                        r['primary']])

    json.dump({'tag': 'pdbev_sedan384',
               'model': 'PD-BEV (Generalizable-BEV / BEVDepth_DG), sedan384 native-384',
               'note': 'CTS_c = NDS(c)/P_TARGET, IMG primary; comp errors not retained',
               'oracle_nds_per_target': ORACLE, 'rows': rows},
              open(osp.join(OUT, 'eval_cts.json'), 'w'), indent=2)

    lines = ['CTS detection eval  (PD-BEV NDS)   tag=pdbev_sedan384   [paper Eq.6]',
             '  numerator P_c = SEDAN384 model deployed on target',
             '  denominator P_TARGET = native-384 target oracle (suv 0.5464, bus 0.6064)',
             '  CTS_c = NDS(c) / P_TARGET   [IMG primary]', '']
    hdr = f'  {"platform":8} {"cond":8} {"NDS":>8} {"mAP6":>8} {"CTS":>8}'
    lines.append(hdr)
    for r in rows:
        star = '  <- primary' if r['primary'] == 'IMG' else ''
        nds_s = f"{r['nds']:.4f}" if r['nds'] is not None else '--'
        m6_s = f"{r['map6']:.4f}" if r['map6'] is not None else '--'
        cts_s = f"{r['cts_nds']:.4f}" if r['cts_nds'] is not None else '--'
        lines.append(f'  {r["platform"]:8} {r["condition"]:8} {nds_s:>8} '
                     f'{m6_s:>8} {cts_s:>8}{star}')
    txt = '\n'.join(lines)
    open(osp.join(OUT, 'eval_cts_summary.txt'), 'w').write(txt + '\n')
    print(txt)
    print(f'\nwrote {OUT}/eval_cts.{{csv,json,summary.txt}}')


if __name__ == '__main__':
    main()
