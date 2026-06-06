"""CTS condition-pkl builder for BEVDepth (BEVDepth-schema analogue of
``build_condition_pkls.make_cts_pkl``).

BEVDepth val info pkls are a **bare list** of per-sample dicts (``mmcv.dump(list)``),
NOT BEVFormer's ``{'infos': [...], 'metadata': {...}}`` wrapper. The DB version is
carried by the exp/evaluator (``self.evaluator.version``), not the pkl. Per-cam
fields live under ``info['cam_infos'][CAM]``:

  - image     : ``['filename']``  (relative to the exp's ``data_root='data/carla'``)
  - extrinsic : ``['calibrated_sensor']['rotation']`` (quat ``[w,x,y,z]``) +
                ``['translation']``  -- the camera MOUNT (cam->ego); this is what
                differs across platforms (sedan z=1.6 vs suv z=2.35 vs bus higher)
  - intrinsic : ``['calibrated_sensor']['camera_intrinsic']``  (identical across
                platforms -> untouched)

GT (``ann_infos``) stays the TARGET's, so the per-target visibility>=2 eval GT is
the target's (matching the seg/CTS definition). sedan<->target join is by parsed
``(scene, frame)`` from ``cam_infos['CAM_FRONT']['filename']`` (per-DB tokens are
independent; scene+frame overlap is 3792/3792). The platform tag is baked into both
the directory and the basename (``RGB-CAM_FRONT/..RGB-subcompact__`` vs
``RGB-suv-CAM_FRONT/..RGB-suv__``), so an image swap copies the whole ``filename``
(string substitution would miss the directory difference).

NEVER writes to BEVFormer's ``data/nuscenes`` pkls -- sources read from
``BEVDepth/data/`` and outputs go wherever the caller points (the CTS driver uses
``bev_det_benchmark/out/cts_<tag>/pkls/``).

CLI smoke:
    python build_condition_pkls_bevdepth.py --target suv --condition IMG --out /tmp/x.pkl
"""
import argparse
import copy
import os
import pickle
import re

BEVDEPTH_DATA = '/home/hanyan_arch/viewpoint/BEVFormer/BEVDepth/data'
SEDAN_VAL = os.path.join(BEVDEPTH_DATA, 'carla_infos_val_sedan.pkl')

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
        pickle.dump(obj, f)          # BARE LIST -- no {'infos'} wrapper


def scene_frame(filename):
    m = SCENE_FRAME_RE.search(filename)
    if not m:
        raise ValueError(f'no scene-/frame- token in: {filename}')
    return m.group(1), m.group(2)


def info_key(sample):
    """(scene, frame) for a BEVDepth sample, parsed from CAM_FRONT's filename.
    Stable across platforms (sedan/suv/bus), unlike per-DB sample_token."""
    return scene_frame(sample['cam_infos']['CAM_FRONT']['filename'])


def build_index(samples):
    return {info_key(s): s for s in samples}


def _copy_extrinsic(dst_cam, src_cam):
    """Overwrite dst cam MOUNT (cam->ego) from src. Only calibrated_sensor
    rotation+translation are platform-dependent; camera_intrinsic is identical
    across platforms and ego_pose is the same trajectory (differs only by the
    per-DB token, which is unused in the single-frame projection) -- both left
    untouched, mirroring BEVFormer's _copy_extrinsic."""
    cs = dst_cam['calibrated_sensor']
    scs = src_cam['calibrated_sensor']
    cs['rotation'] = list(scs['rotation'])
    cs['translation'] = list(scs['translation'])


def make_cts_pkl(condition, target, out_path, sedan_pkl=SEDAN_VAL,
                 target_pkl=None):
    """Write one CTS condition pkl based on the TARGET eval set.

    base = target pkl (target GT/visibility/DB). Each condition swaps the cam
    image and/or mount extrinsic toward the SEDAN source:
        NORMAL = sedan img + sedan ext   EXT = sedan img + target ext
        IMG    = target img + sedan ext   CAL = target img + target ext (as-is)
    Returns (n, matched, missing).
    """
    assert condition in CTS_CONDITIONS, condition
    img_src, ext_src = CTS_CONDITIONS[condition]

    if target_pkl is None:
        target_pkl = os.path.join(BEVDEPTH_DATA,
                                  f'carla_infos_val_{target}.pkl')
    target_data = load_pkl(target_pkl)            # bare list; target GT/version
    if (img_src, ext_src) == ('t', 't'):          # CAL == target pkl as-is
        dump_pkl(target_data, out_path)
        n = len(target_data)
        return n, n, 0

    sidx = build_index(load_pkl(sedan_pkl))
    new = copy.deepcopy(target_data)              # keeps target GT/ann_infos
    matched = missing = 0
    for sample in new:
        ssample = sidx.get(info_key(sample))
        if ssample is None:
            missing += 1
            continue
        matched += 1
        for cam_name, cam in sample['cam_infos'].items():
            scam = ssample['cam_infos'][cam_name]
            if img_src == 's':                    # swap image toward sedan
                cam['filename'] = scam['filename']
            if ext_src == 's':                    # swap mount toward sedan
                _copy_extrinsic(cam, scam)
    dump_pkl(new, out_path)
    return len(new), matched, missing


# --------------------------------------------------------------------------- #
# VP  -- viewpoint robustness (carla_VR), BEVDepth schema
# --------------------------------------------------------------------------- #
# carla_VR ships 31 variants = baseline + 10 each for yaw/pitch/roll, signed
# +/-{4,8,12,16,20}. Per (scene, cam, variant) it provides the perturbed image
# AND the perturbed extrinsic in BOTH conventions: sensor2lidar_* (BEVFormer)
# and sensor2ego_* (BEVDepth). It has NO boxes -> GT always comes from the sedan
# val pkl. BEVDepth lifts the frustum to the EGO frame, so we perturb the cam->ego
# extrinsic = calibrated_sensor, using the RAW sensor2ego quaternion (NO matrix
# conversion, and NEVER the sensor2lidar values -- they differ by the lidar mount
# height in z and would shift the camera vertically).
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
    Absolute so BEVDepth's os.path.join(data_root, filename) ignores data_root."""
    # carla_geobev (val GT/images) is a 1/2-rate RELABEL of the carla_VR capture:
    # geobev frame N is the SAME world-moment as carla_VR frame 2N (verified --
    # geobev 0269/0150 == VR baseline 0300, exact 2.000x across clean-motion
    # scenes). Joining by the bare frame number loads a DIFFERENT scene moment, so
    # every image-swapped VR/CR condition was fed mismatched images -> use 2N.
    return (f"{VR_ROOT}/sweeps/RGB-{cam}_{variant}/"
            f"SimBEV-scene-{scene}-frame-{int(frame) * 2:04d}-RGB-{cam}_{variant}.jpg")


def variant_extrinsic_ego(scene, cam, variant, meta=None):
    """(sensor2ego_rotation [w,x,y,z], sensor2ego_translation xyz) for a variant
    -- the RAW quaternion BEVDepth's calibrated_sensor expects (no conversion).
    metadata scene keys use the underscore form (scene_0220)."""
    meta = meta or load_vr_metadata()
    rec = meta['scenes'][f'scene_{scene}'][cam][variant]
    return list(rec['sensor2ego_rotation']), list(rec['sensor2ego_translation'])


def make_vp_infos_bevdepth(base_infos, condition, axis, signed_mag, protocol):
    """Deep-copied BEVDepth infos (bare list) with VP swaps applied in-memory.

    protocol = 'all'  -> perturb all 6 cams together
    protocol = <CAM>  -> perturb only that camera (per-cam), others baseline.
    'Normal' returns an untouched deep copy (the oracle subset).
    Image swap -> cam_infos[cam]['filename'] = VR variant path (baseline ext kept).
    Extrinsic swap -> calibrated_sensor rotation/translation = variant sensor2ego.
    """
    swap_img, swap_ext = VP_CONDITIONS[condition]
    variant = variant_key(axis, signed_mag)
    target_cams = CAM_NAMES if protocol == 'all' else [protocol]
    meta = load_vr_metadata()

    new = copy.deepcopy(base_infos)
    for sample in new:
        scene, frame = info_key(sample)
        for cam_name in target_cams:
            cam = sample['cam_infos'][cam_name]
            if swap_img:
                cam['filename'] = vr_image_path(scene, frame, cam_name, variant)
            if swap_ext:
                rot, trans = variant_extrinsic_ego(scene, cam_name, variant, meta)
                cam['calibrated_sensor']['rotation'] = rot
                cam['calibrated_sensor']['translation'] = trans
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=['suv', 'bus'])
    ap.add_argument('--condition', required=True, choices=list(CTS_CONDITIONS))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    n, matched, missing = make_cts_pkl(args.condition, args.target, args.out)
    print(f'[cts-bevdepth] {args.target} {args.condition}: n={n} '
          f'matched={matched} missing={missing} -> {args.out}')


if __name__ == '__main__':
    main()
