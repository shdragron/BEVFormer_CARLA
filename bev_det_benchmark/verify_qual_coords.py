"""Lock down the coordinate recipe for CONDITION-AWARE qual rendering.

Qual draws EGO-frame boxes via pixel = K @ inv(sensor2ego) @ corners.
VP condition pkls perturb ONLY sensor2lidar_* (sensor2ego_* left stale), while the
model's eval uses lidar2img from the perturbed sensor2lidar. So to render a VP
ER/CR/CTS frame the way the MODEL saw it, we must derive the condition's sensor2ego
from its sensor2lidar via the fixed mount transform lidar2ego (constant, camera-
perturbation-independent):

    lidar2ego          = sensor2ego_base @ inv(sensor2lidar_base)     (per cam, SE3)
    sensor2ego_cond    = lidar2ego @ sensor2lidar_cond

This script proves (1) lidar2ego is ~identity-rotation + [0,0,1.8] and constant
across cams, (2) VP swaps leave sensor2ego stale (naive renderer would be wrong for
ER/CR), (3) the derived sensor2ego for VR (image-only) reduces to baseline (renderer
unchanged), and (4) for CR the derived sensor2ego differs from the stale field.
"""
import os, pickle, sys
import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_condition_pkls as B

DATA = '/home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes'
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def s2l_SE3(cam):
    """sensor2lidar as 4x4 (rotation stored as 3x3 matrix in the pkl)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(cam['sensor2lidar_rotation'], float)
    T[:3, 3] = np.asarray(cam['sensor2lidar_translation'], float)
    return T


def s2e_SE3(cam):
    """sensor2ego as 4x4 (rotation stored as [w,x,y,z] quaternion in the pkl)."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(cam['sensor2ego_rotation']).rotation_matrix
    T[:3, 3] = np.asarray(cam['sensor2ego_translation'], float)
    return T


def main():
    sedan = pickle.load(open(f'{DATA}/sedan_infos_val.pkl', 'rb'))['infos']
    info = sedan[0]
    print(f"sample token={info['token']}  scene/frame={B.info_key(info)}\n")

    # (1) lidar2ego per cam from baseline
    print("[1] lidar2ego = sensor2ego_base @ inv(sensor2lidar_base)  (expect R~I, t~[0,0,1.8])")
    l2e_list = []
    for cam in CAMS:
        c = info['cams'][cam]
        l2e = s2e_SE3(c) @ np.linalg.inv(s2l_SE3(c))
        l2e_list.append(l2e)
        rot_dev = np.abs(l2e[:3, :3] - np.eye(3)).max()
        print(f"   {cam:16s} t={np.round(l2e[:3,3],4)}  max|R-I|={rot_dev:.2e}")
    l2e = l2e_list[0]
    spread = max(np.abs(x - l2e).max() for x in l2e_list)
    print(f"   -> cross-cam spread of lidar2ego = {spread:.2e} (expect ~0: same mount)\n")

    # (2) build CR (both) + VR (img-only) pitch+20 all-cam condition infos
    base = [info]
    cr = B.make_vp_infos(base, 'CR', 'pitch', 20, 'all')[0]
    vr = B.make_vp_infos(base, 'VR', 'pitch', 20, 'all')[0]
    er = B.make_vp_infos(base, 'ER', 'pitch', 20, 'all')[0]

    print("[2] VP swaps leave sensor2ego STALE (== baseline) though sensor2lidar changed:")
    for cond, ci in (('CR', cr), ('ER', er)):
        c = ci['cams']['CAM_FRONT']
        s2l_changed = np.abs(np.asarray(c['sensor2lidar_rotation'], float)
                             - np.asarray(info['cams']['CAM_FRONT']['sensor2lidar_rotation'], float)).max()
        s2e_changed = np.abs(Quaternion(c['sensor2ego_rotation']).rotation_matrix
                             - Quaternion(info['cams']['CAM_FRONT']['sensor2ego_rotation']).rotation_matrix).max()
        print(f"   {cond}: max|s2l_rot - base| = {s2l_changed:.3e} (perturbed)   "
              f"max|s2e_rot - base| = {s2e_changed:.3e} (STALE if ~0)")

    # (3) derived sensor2ego_cond = lidar2ego @ sensor2lidar_cond
    print("\n[3] derived sensor2ego_cond vs (a) stale field, (b) eval projection:")
    for cond, ci in (('VR', vr), ('ER', er), ('CR', cr)):
        cam = 'CAM_FRONT'
        c = ci['cams'][cam]
        l2e_cam = s2e_SE3(info['cams'][cam]) @ np.linalg.inv(s2l_SE3(info['cams'][cam]))
        s2e_derived = l2e_cam @ s2l_SE3(c)
        s2e_stale = s2e_SE3(c)
        diff = np.abs(s2e_derived - s2e_stale).max()
        tag = 'reduces to baseline (img-only)' if cond == 'VR' else 'differs from stale -> MUST use derived'
        print(f"   {cond}: max|derived - stale s2e| = {diff:.3e}  [{tag}]")

    # (4) cross-check: derived ego->cam projection must equal eval's lidar->cam on a
    #     test point (ego box-centre p_ego; p_lidar = inv(lidar2ego) @ p_ego)
    print("\n[4] projection equivalence (ego-frame via derived s2e == lidar-frame via s2l), CR:")
    cam = 'CAM_FRONT'
    c = cr['cams'][cam]
    K = np.asarray(c['cam_intrinsic'], float)
    l2e_cam = s2e_SE3(info['cams'][cam]) @ np.linalg.inv(s2l_SE3(info['cams'][cam]))
    s2e_derived = l2e_cam @ s2l_SE3(c)
    p_ego = np.array([8.0, 1.5, 0.8, 1.0])              # a plausible box centre in ego
    p_lidar = np.linalg.inv(l2e_cam) @ p_ego
    # ego path
    pc_e = np.linalg.inv(s2e_derived) @ p_ego
    uv_e = K @ pc_e[:3]; uv_e = uv_e[:2] / uv_e[2]
    # lidar path (what the model uses)
    pc_l = np.linalg.inv(s2l_SE3(c)) @ p_lidar
    uv_l = K @ pc_l[:3]; uv_l = uv_l[:2] / uv_l[2]
    print(f"   ego-path uv ={np.round(uv_e,3)}   lidar-path uv ={np.round(uv_l,3)}   "
          f"max|diff|={np.abs(uv_e-uv_l).max():.2e}")
    print("\nDONE.")


if __name__ == '__main__':
    main()
