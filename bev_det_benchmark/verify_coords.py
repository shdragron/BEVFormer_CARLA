"""Regression test for the coordinate-system invariants this benchmark relies on.

bev_det_benchmark reproduces the seg (CVT) robustness study by *surgically
swapping cam fields in the val .pkl* (data_path for image swaps,
sensor2lidar_* for extrinsic swaps) and running the stock model unchanged
(build_condition_pkls.py). That is only valid if a set of coordinate facts
hold. This script asserts every one of them against the real data and exits
non-zero if any breaks, so a regenerated pkl / metadata can be re-checked in one
command:

    conda activate bevformer-b200
    python bev_det_benchmark/verify_coords.py        # exit 0 = all invariants hold

Checks (all relative to the BEVFormer pkl convention the model actually consumes
in CustomNuScenesDataset.get_data_info -> lidar2img from sensor2lidar+intrinsic):

  A. VP baseline match : Quaternion(meta baseline quat).rotation_matrix == sedan
     pkl sensor2lidar_rotation (+ translations), for all 6 cams across scenes.
     This is THE anchor: it proves viewpoint_metadata.json's sensor2lidar
     convention is identical to the pkl's, so VP extrinsic swaps (overwrite with
     the variant value) apply the perturbation in the correct frame.
  B. scene coverage    : every sedan-val scene exists in the metadata (else ER/CR
     would KeyError in build_condition_pkls.variant_extrinsic).
  C. intra-scene const : sensor2lidar is constant across frames within a scene,
     so the per-(scene,cam,variant) metadata loses no per-frame information.
  D. intrinsic invariant: cam_intrinsic is identical across platforms, so the CTS
     extrinsic-only swap (which never touches intrinsics) is physically correct.
  E. CTS join          : sedan<->suv<->bus join by (scene,frame) is complete and
     gt_boxes / gt_names are bit-identical across platforms (valid_flag differs by
     design: visibility changes with mount height).
  F. VR perturbation    : a yaw/pitch/roll ±mag variant is exactly that rotation
     magnitude about the expected camera-local axis (yaw->y, pitch->x, roll->z).
"""
import argparse
import json
import pickle
import re
import sys

import numpy as np
from pyquaternion import Quaternion

# --------------------------------------------------------------------------- #
# config (defaults match the repo layout; override for a regenerated dataset)
# --------------------------------------------------------------------------- #
DATA_ROOT = '/home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes'
VR_META = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_VR/viewpoint_metadata.json'
CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
SCENE_FRAME_RE = re.compile(r'scene-(\d+)-frame-(\d+)')

# tolerances
TOL_EXACT = 1e-6     # quat->matrix vs pkl, intra-scene constancy
TOL_INTRIN = 1e-9    # intrinsics must be bit-equal across platforms
TOL_ANGLE = 0.5      # deg, VR perturbation magnitude
N_SCENES = 10        # scenes sampled for the per-(scene,cam) checks
N_JOIN = 300         # matched samples checked for gt identity


def load_infos(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['infos']


def key(info):
    m = SCENE_FRAME_RE.search(info['cams']['CAM_FRONT']['data_path'])
    if not m:
        raise ValueError(f"no scene-/frame- token in {info['cams']['CAM_FRONT']['data_path']}")
    return m.group(1), m.group(2)


def geodesic_deg(q_from, q_to):
    """Rotation magnitude (deg, in [0,180]) and unit axis from q_from to q_to."""
    rel = q_from.inverse * q_to
    ang = np.degrees(2.0 * np.arccos(min(1.0, abs(rel.w))))
    return ang, rel.axis


class Report:
    def __init__(self):
        self.failed = []

    def check(self, name, ok, detail):
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] {name}: {detail}")
        if not ok:
            self.failed.append(name)

    def done(self):
        print('\n' + ('=' * 64))
        if self.failed:
            print(f"RESULT: FAIL ({len(self.failed)} check(s) failed: {', '.join(self.failed)})")
            return 1
        print('RESULT: PASS (all coordinate invariants hold)')
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--vr-meta', default=VR_META)
    args = ap.parse_args()

    print(f"Loading pkls + metadata from\n  {args.data_root}\n  {args.vr_meta}")
    sedan = load_infos(f'{args.data_root}/sedan_infos_val.pkl')
    suv = load_infos(f'{args.data_root}/suv_infos_val.pkl')
    bus = load_infos(f'{args.data_root}/bus_infos_val.pkl')
    with open(args.vr_meta) as f:
        meta = json.load(f)
    print(f"  sedan={len(sedan)} suv={len(suv)} bus={len(bus)} infos\n")

    sidx = {key(i): i for i in sedan}
    suvidx = {key(i): i for i in suv}
    busidx = {key(i): i for i in bus}
    by_scene = {}
    for i in sedan:
        by_scene.setdefault(key(i)[0], []).append(i)
    sample_scenes = sorted(by_scene)[:N_SCENES]

    R = Report()

    # --- A. VP baseline match ------------------------------------------------ #
    print("[A] VP baseline (yaw0pitch0roll0): meta quat->matrix vs sedan pkl sensor2lidar")
    max_rot = max_trans = 0.0
    n = 0
    for sc in sample_scenes:
        skey = f'scene_{sc}'
        if skey not in meta['scenes']:
            continue
        info = by_scene[sc][0]
        for cam in CAM_NAMES:
            rec = meta['scenes'][skey][cam]['yaw0pitch0roll0']
            meta_rot = Quaternion(rec['sensor2lidar_rotation']).rotation_matrix
            meta_t = np.asarray(rec['sensor2lidar_translation'], float)
            pkl_rot = np.asarray(info['cams'][cam]['sensor2lidar_rotation'], float)
            pkl_t = np.asarray(info['cams'][cam]['sensor2lidar_translation'], float)
            max_rot = max(max_rot, np.abs(meta_rot - pkl_rot).max())
            max_trans = max(max_trans, np.abs(meta_t - pkl_t).max())
            n += 1
    R.check('A_baseline_extrinsic', max_rot < TOL_EXACT and max_trans < TOL_EXACT,
            f"{n} (scene,cam) pairs; max|rot|={max_rot:.2e} max|trans|={max_trans:.2e} (tol {TOL_EXACT})")

    # --- B. scene coverage --------------------------------------------------- #
    print("[B] sedan-val scenes all present in metadata (else ER/CR KeyError)")
    sedan_scenes = {key(i)[0] for i in sedan}
    meta_scenes = {k.replace('scene_', '') for k in meta['scenes']}
    missing = sorted(sedan_scenes - meta_scenes)
    R.check('B_scene_coverage', not missing,
            f"sedan={len(sedan_scenes)} meta={len(meta_scenes)} missing={missing or 'NONE'}")

    # --- C. intra-scene sensor2lidar constancy ------------------------------- #
    print("[C] sensor2lidar constant across frames within a scene")
    max_intra = 0.0
    worst = None
    for sc in sample_scenes:
        infos = by_scene[sc]
        if len(infos) < 2:
            continue
        ref = infos[0]
        for cam in CAM_NAMES:
            r0 = np.asarray(ref['cams'][cam]['sensor2lidar_rotation'], float)
            t0 = np.asarray(ref['cams'][cam]['sensor2lidar_translation'], float)
            for i in infos[1:]:
                d = max(np.abs(np.asarray(i['cams'][cam]['sensor2lidar_rotation'], float) - r0).max(),
                        np.abs(np.asarray(i['cams'][cam]['sensor2lidar_translation'], float) - t0).max())
                if d > max_intra:
                    max_intra, worst = d, (sc, cam)
    R.check('C_intra_scene_constant', max_intra < TOL_EXACT,
            f"max variation={max_intra:.2e} worst={worst} (tol {TOL_EXACT})")

    # --- D. cam_intrinsic invariant across platforms ------------------------- #
    print("[D] cam_intrinsic identical sedan vs suv vs bus (CTS keeps intrinsic)")
    max_di = 0.0
    ncmp = 0
    for k, sinfo in list(sidx.items())[:N_JOIN]:
        for tgt in (suvidx, busidx):
            tinfo = tgt.get(k)
            if tinfo is None:
                continue
            for cam in CAM_NAMES:
                d = np.abs(np.asarray(sinfo['cams'][cam]['cam_intrinsic'], float)
                           - np.asarray(tinfo['cams'][cam]['cam_intrinsic'], float)).max()
                max_di = max(max_di, d)
            ncmp += 1
    R.check('D_intrinsic_invariant', max_di < TOL_INTRIN,
            f"{ncmp} samples; max|intrinsic diff|={max_di:.2e} (tol {TOL_INTRIN})")

    # --- E. CTS join completeness + gt identity ------------------------------ #
    print("[E] CTS (scene,frame) join complete + gt_boxes/gt_names bit-identical")
    ok_join = True
    for tgtidx, name in ((suvidx, 'suv'), (busidx, 'bus')):
        common = set(sidx) & set(tgtidx)
        only_s, only_t = set(sidx) - set(tgtidx), set(tgtidx) - set(sidx)
        gt_mis = nm_mis = 0
        for k in list(common)[:N_JOIN]:
            sb = np.asarray(sidx[k]['gt_boxes'], float)
            tb = np.asarray(tgtidx[k]['gt_boxes'], float)
            if sb.shape != tb.shape or not np.allclose(sb, tb, atol=1e-6):
                gt_mis += 1
            if list(sidx[k]['gt_names']) != list(tgtidx[k]['gt_names']):
                nm_mis += 1
        passed = (not only_s and not only_t and gt_mis == 0 and nm_mis == 0)
        ok_join = ok_join and passed
        print(f"    sedan<->{name}: common={len(common)} sedan_only={len(only_s)} "
              f"{name}_only={len(only_t)} box_mismatch={gt_mis} name_mismatch={nm_mis}")
    R.check('E_cts_join_and_gt', ok_join, 'see per-platform line above')

    # --- F. VR perturbation magnitude/axis ----------------------------------- #
    print("[F] VR perturbation = requested deg about expected cam-local axis")
    # yaw->y(index1), pitch->x(index0), roll->z(index2)
    AXIS_IDX = {'yaw': 1, 'pitch': 0, 'roll': 2}
    sc = next(s for s in sample_scenes if f'scene_{s}' in meta['scenes'])
    skey = f'scene_{sc}'
    worst_ang_err = 0.0
    worst_axis_leak = 0.0
    nF = 0
    for cam in CAM_NAMES:
        base_q = Quaternion(meta['scenes'][skey][cam]['yaw0pitch0roll0']['sensor2lidar_rotation'])
        for axis, ai in AXIS_IDX.items():
            for mag in (4, 20):
                v = {'yaw': 0, 'pitch': 0, 'roll': 0}
                v[axis] = mag
                vk = f"yaw{v['yaw']}pitch{v['pitch']}roll{v['roll']}"
                if vk not in meta['scenes'][skey][cam]:
                    continue
                q = Quaternion(meta['scenes'][skey][cam][vk]['sensor2lidar_rotation'])
                ang, ax = geodesic_deg(base_q, q)
                worst_ang_err = max(worst_ang_err, abs(ang - mag))
                # expected axis component dominant, others ~0
                leak = max(abs(ax[j]) for j in range(3) if j != ai)
                worst_axis_leak = max(worst_axis_leak, leak)
                nF += 1
    R.check('F_vr_perturbation', worst_ang_err < TOL_ANGLE and worst_axis_leak < 0.05,
            f"{nF} variants; max|angle-mag|={worst_ang_err:.3f}deg (tol {TOL_ANGLE}) "
            f"max off-axis leak={worst_axis_leak:.3f} (tol 0.05)")

    return R.done()


if __name__ == '__main__':
    sys.exit(main())
