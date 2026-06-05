"""CTS + VP condition-pkl builder for BEVDet (BEVDet-schema analogue of
``build_condition_pkls`` / ``build_condition_pkls_bevdepth``).

BEVDet val info pkls (``BEVDet/tools/create_carla_data_bevdet.py``) use the
BEVFormer-style wrapper ``{'infos': [...], 'metadata': {'version':
'v1.0-carla_<veh>_eval'}}`` and store GT as ``info['ann_infos'] = (gt_boxes,
gt_labels)``.  Per-cam fields live directly under ``info['cams'][CAM]``:

  - image     : ``['data_path']``  (relative ``data/nuscenes/sweeps/...``)
  - extrinsic : ``['sensor2ego_rotation']`` (quat ``[w,x,y,z]``) +
                ``['sensor2ego_translation']`` -- the camera MOUNT (cam->ego),
                which is what BEVDet's ``PrepareImageInputs`` reads (it lifts the
                frustum to the ego frame) and what differs across platforms
                (sedan/suv/bus mount heights).  NOT ``sensor2lidar_*`` -- those are
                only used by the lidar sweep loader and would shift z by the lidar
                mount height.
  - intrinsic : ``['cam_intrinsic']``  (identical across platforms -> untouched)

GT (``ann_infos``) stays the TARGET's, so the per-target visibility>=2 eval GT is
the target's (matching the CTS/seg definition).  sedan<->target join is by parsed
``(scene, frame)`` from ``cams['CAM_FRONT']['data_path']`` (per-DB tokens are
independent; scene+frame overlap is 3792/3792).

NEVER writes to BEVFormer's ``data/nuscenes`` pkls -- sources read from
``BEVDet/data/bevdet_infos/`` and outputs go wherever the caller points (the CTS
driver uses ``bev_det_benchmark/out/cts_<tag>/pkls/``).

CLI smoke:
    python build_condition_pkls_bevdet.py --target suv --condition IMG --out /tmp/x.pkl
"""
import argparse
import copy
import os
import pickle
import re

import numpy as np

BEVDET_ROOT = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDet'
BEVDET_DATA = os.path.join(BEVDET_ROOT, 'data', 'bevdet_infos')
SEDAN_VAL = os.path.join(BEVDET_DATA, 'sedan_infos_val.pkl')

SCENE_FRAME_RE = re.compile(r'scene-(\d+)-frame-(\d+)')

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# condition: (image_source, extrinsic_source)   's'=sedan, 't'=target
CTS_CONDITIONS = {
    'NORMAL': ('s', 's'),   # sedan img + sedan ext (sedan-inputs reference)
    'EXT':    ('s', 't'),   # sedan img + target ext
    'IMG':    ('t', 's'),   # target img + sedan ext  (primary)
    'CAL':    ('t', 't'),   # target img + target ext (== target pkl as-is)
}


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pkl(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)          # {'infos': [...], 'metadata': {...}} wrapper


def scene_frame(data_path):
    m = SCENE_FRAME_RE.search(data_path)
    if not m:
        raise ValueError(f'no scene-/frame- token in: {data_path}')
    return m.group(1), m.group(2)


def info_key(info):
    """(scene, frame) for a BEVDet sample, parsed from CAM_FRONT's data_path.
    Stable across platforms (sedan/suv/bus), unlike the per-DB sample token."""
    return scene_frame(info['cams']['CAM_FRONT']['data_path'])


def build_index(infos):
    return {info_key(i): i for i in infos}


def _copy_extrinsic(dst_cam, src_cam):
    """Overwrite dst cam MOUNT (cam->ego) from src. Only sensor2ego_* is
    platform-dependent and read by PrepareImageInputs; cam_intrinsic is identical
    across platforms (untouched). sensor2lidar_* is copied too for self-consistency
    (unused by the camera path) -- mirrors BEVFormer's _copy_extrinsic."""
    dst_cam['sensor2ego_rotation'] = list(src_cam['sensor2ego_rotation'])
    dst_cam['sensor2ego_translation'] = list(src_cam['sensor2ego_translation'])
    if 'sensor2lidar_rotation' in src_cam:
        dst_cam['sensor2lidar_rotation'] = np.array(src_cam['sensor2lidar_rotation'])
        dst_cam['sensor2lidar_translation'] = np.array(src_cam['sensor2lidar_translation'])


def make_cts_pkl(condition, target, out_path, sedan_pkl=SEDAN_VAL,
                 target_pkl=None):
    """Write one CTS condition pkl based on the TARGET eval set.

    base = target pkl (target GT/ann_infos/visibility + metadata.version). Each
    condition swaps the cam image and/or mount extrinsic toward the SEDAN source:
        NORMAL = sedan img + sedan ext   EXT = sedan img + target ext
        IMG    = target img + sedan ext   CAL = target img + target ext (as-is)
    Returns (n, matched, missing).
    """
    assert condition in CTS_CONDITIONS, condition
    img_src, ext_src = CTS_CONDITIONS[condition]

    if target_pkl is None:
        target_pkl = os.path.join(BEVDET_DATA, f'{target}_infos_val.pkl')
    target_data = load_pkl(target_pkl)            # {'infos':[...], 'metadata':{...}}
    if (img_src, ext_src) == ('t', 't'):          # CAL == target pkl as-is
        dump_pkl(target_data, out_path)
        n = len(target_data['infos'])
        return n, n, 0

    sidx = build_index(load_pkl(sedan_pkl)['infos'])
    new = copy.deepcopy(target_data)              # keeps target ann_infos/version
    matched = missing = 0
    for info in new['infos']:
        sinfo = sidx.get(info_key(info))
        if sinfo is None:
            missing += 1
            continue
        matched += 1
        for cam_name, cam in info['cams'].items():
            scam = sinfo['cams'][cam_name]
            if img_src == 's':                    # swap image toward sedan
                cam['data_path'] = scam['data_path']
            if ext_src == 's':                    # swap mount toward sedan
                _copy_extrinsic(cam, scam)
    dump_pkl(new, out_path)
    return len(new['infos']), matched, missing


# --------------------------------------------------------------------------- #
# VP  -- viewpoint robustness (carla_VR), BEVDet schema
# --------------------------------------------------------------------------- #
# carla_VR ships 31 variants = baseline + 10 each for yaw/pitch/roll, signed
# +/-{4,8,12,16,20}. Per (scene, cam, variant) it provides the perturbed image AND
# the perturbed extrinsic in BOTH conventions (sensor2lidar_* and sensor2ego_*); it
# has NO boxes -> GT always comes from the sedan val pkl. BEVDet lifts the frustum
# to the EGO frame, so we perturb the cam->ego extrinsic = sensor2ego_*, using the
# RAW sensor2ego quaternion (NO matrix conversion, NEVER sensor2lidar_*).
VR_ROOT = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_VR'
VR_META = os.path.join(VR_ROOT, 'viewpoint_metadata.json')
VP_AXES = ['yaw', 'pitch', 'roll']
VP_MAGNITUDES = [-20, -16, -12, -8, -4, 4, 8, 12, 16, 20]   # signed, per axis
VP_CONDITIONS = {                       # condition: (swap_image, swap_extrinsic)
    'Normal': (False, False),           # baseline == stock sedan subset (oracle)
    'ER':     (False, True),            # extrinsic-only
    'VR':     (True,  False),           # image-only  (primary)
    'CR':     (True,  True),            # both
}

_VR_META = None


def load_vr_metadata():
    """Cached viewpoint_metadata.json (scenes[scene_us][cam][variant] -> extr)."""
    global _VR_META
    if _VR_META is None:
        import json
        with open(VR_META) as f:
            _VR_META = json.load(f)
    return _VR_META


def variant_key(axis, signed_mag):
    """('yaw', 20) -> 'yaw20pitch0roll0';  ('pitch', -8) -> 'yaw0pitch-8roll0'."""
    vals = {'yaw': 0, 'pitch': 0, 'roll': 0}
    vals[axis] = signed_mag
    return f"yaw{vals['yaw']}pitch{vals['pitch']}roll{vals['roll']}"


def vr_image_path(scene, frame, cam, variant):
    """ABSOLUTE carla_VR variant-image path (filenames use the dash scene form).
    Absolute so BEVDet's mmcv.imread(data_path) reads it regardless of cwd."""
    return (f"{VR_ROOT}/sweeps/RGB-{cam}_{variant}/"
            f"SimBEV-scene-{scene}-frame-{int(frame):04d}-RGB-{cam}_{variant}.jpg")


def variant_extrinsic_ego(scene, cam, variant, meta=None):
    """(sensor2ego_rotation [w,x,y,z], sensor2ego_translation xyz) for a variant
    -- the RAW quaternion BEVDet's sensor2ego_rotation field expects (no conversion).
    metadata scene keys use the underscore form (scene_0220)."""
    meta = meta or load_vr_metadata()
    rec = meta['scenes'][f'scene_{scene}'][cam][variant]
    return list(rec['sensor2ego_rotation']), list(rec['sensor2ego_translation'])


def make_vp_infos(base_infos, condition, axis, signed_mag, protocol):
    """Deep-copied BEVDet infos LIST with VP swaps applied in-memory.

    base_infos = the ``['infos']`` list of the sedan val pkl (subset).
    protocol = 'all'  -> perturb all 6 cams together
    protocol = <CAM>  -> perturb only that camera (per-cam), others baseline.
    'Normal' returns an untouched deep copy (the oracle subset).
    Image swap -> cams[cam]['data_path'] = VR variant jpg (baseline ext kept).
    Extrinsic swap -> cams[cam]['sensor2ego_*'] = variant sensor2ego.
    """
    swap_img, swap_ext = VP_CONDITIONS[condition]
    variant = variant_key(axis, signed_mag)
    target_cams = CAM_NAMES if protocol == 'all' else [protocol]
    meta = load_vr_metadata()

    new = copy.deepcopy(base_infos)
    for info in new:
        scene, frame = info_key(info)
        for cam_name in target_cams:
            cam = info['cams'][cam_name]
            if swap_img:
                cam['data_path'] = vr_image_path(scene, frame, cam_name, variant)
            if swap_ext:
                rot, trans = variant_extrinsic_ego(scene, cam_name, variant, meta)
                cam['sensor2ego_rotation'] = rot
                cam['sensor2ego_translation'] = trans
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=['suv', 'bus'])
    ap.add_argument('--condition', required=True, choices=list(CTS_CONDITIONS))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    n, matched, missing = make_cts_pkl(args.condition, args.target, args.out)
    print(f'[cts-bevdet] {args.target} {args.condition}: n={n} '
          f'matched={matched} missing={missing} -> {args.out}')


if __name__ == '__main__':
    main()
