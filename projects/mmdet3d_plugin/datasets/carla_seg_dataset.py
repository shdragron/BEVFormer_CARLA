"""CARLA vehicle-occupancy BEV segmentation dataset (GaussianLSS GT).

Reads the GaussianLSS-format scene jsons directly (no nuScenes DB), yields the
data dict BEVFormer's seg pipeline expects: 6x 224x480 images, lidar2img per
cam, and a 200x200 binary vehicle-occupancy mask in the BEVFormer grid layout.

Verified facts (see bevformer_seg/NOTES.md, locked 2026-06-14):
  - vehicle = (bev_png >> 4) & 1   (bit4; bit0/1 are road)
  - lidar2img = viewpad(I_rescaled) @ E   (E = ego2cam; rescale = resize
    1600x900->480x270 then top_crop 46, baked into intrinsic)
  - GT layout to BEVFormer grid: mask[::-1, ::-1].T
"""
import json, os
import numpy as np
import mmcv
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
SRC_W, SRC_H = 1600.0, 900.0
RS_W, RS_H = 480.0, 270.0
TOP_CROP = 46
VEH_BIT = 4
BEV = 200


@DATASETS.register_module()
class CarlaSegDataset(Dataset):
    CLASSES = ('vehicle',)

    def __init__(self, labels_dir, image_root, split_file, pipeline,
                 bev_size=200, test_mode=False, min_visibility=2, **kwargs):
        self.labels_dir = labels_dir
        self.image_root = image_root
        self.bev = bev_size
        self.test_mode = test_mode
        self.min_visibility = min_visibility
        if split_file:
            scenes = [s.strip() for s in open(split_file) if s.strip()]
        else:                                  # all scene jsons in labels_dir
            import glob
            scenes = sorted(os.path.basename(p)[:-5] for p in
                            glob.glob(os.path.join(labels_dir, 'scene_*.json')))
        self.frames = []
        for s in scenes:
            jp = os.path.join(labels_dir, f'{s}.json')
            if not os.path.exists(jp):
                continue
            for fi, fr in enumerate(json.load(open(jp))):
                self.frames.append((s, fi, fr))
        self.flag = np.zeros(len(self.frames), dtype=np.uint8)  # mmdet sampler
        from mmdet.datasets.pipelines import Compose
        self.pipeline = Compose(pipeline)
        print(f'[CarlaSeg] {labels_dir.split("/")[-1]}: {len(scenes)} scenes -> '
              f'{len(self.frames)} frames')

    def __len__(self):
        return len(self.frames)

    @staticmethod
    def _rescale_K(I):
        I = np.array(I, dtype=np.float64).copy()
        I[0, 0] *= RS_W / SRC_W; I[0, 2] *= RS_W / SRC_W
        I[1, 1] *= RS_H / SRC_H; I[1, 2] *= RS_H / SRC_H
        I[1, 2] -= TOP_CROP
        return I

    def _veh_mask(self, scene, bev_png):
        from PIL import Image
        arr = np.array(Image.open(os.path.join(self.labels_dir, scene, bev_png)))
        g = ((arr >> VEH_BIT) & 1).astype(np.uint8)          # bit4 = vehicle
        return g[::-1, ::-1].T.copy()                        # -> BEVFormer grid

    def _gt_mask(self, scene, fr):
        """Visibility-filtered GT, 3DOD-style: keep only vehicle cells with
        visibility>=min_visibility; low-vis vehicles become background (so a
        prediction there is a false positive, exactly like the detection
        models' visibility>=2 GT removal). Returned in the BEVFormer grid."""
        from PIL import Image
        arr = np.array(Image.open(os.path.join(self.labels_dir, scene, fr['bev'])))
        veh = ((arr >> VEH_BIT) & 1).astype(np.uint8)        # bit4 = vehicle
        vis = np.array(Image.open(os.path.join(self.labels_dir, scene, fr['visibility'])))
        veh = veh & (vis >= self.min_visibility).astype(np.uint8)
        return veh[::-1, ::-1].T.copy()                      # -> BEVFormer grid

    def get_data_info(self, index):
        scene, fi, fr = self.frames[index]
        l2i = []
        for i in range(6):
            K4 = np.eye(4); K4[:3, :3] = self._rescale_K(fr['intrinsics'][i])
            E = np.array(fr['extrinsics'][i], dtype=np.float64)   # ego2cam
            l2i.append(K4 @ E)
        img_filenames = [os.path.join(self.image_root, fr['images'][i]) for i in range(6)]
        can_bus = np.zeros(18, dtype=np.float64)
        # 3DOD-style visibility filtering baked into the GT (low-vis vehicles ->
        # background), so training and eval both match the detection protocol.
        gt_seg = self._gt_mask(scene, fr)                         # (200,200) uint8
        return dict(
            sample_idx=index,
            img_filename=img_filenames,
            lidar2img=l2i,
            can_bus=can_bus,
            scene_token=fr['token'],          # unique per frame -> prev_bev=None
            prev_idx='', next_idx='',
            gt_seg=gt_seg,
            img_shape_target=(int(RS_H - TOP_CROP), int(RS_W)),   # (224,480)
        )

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare(idx)
        while True:
            data = self.prepare(idx)
            if data is None:
                idx = (idx + 1) % len(self)
                continue
            return data

    def prepare(self, index):
        input_dict = self.get_data_info(index)
        input_dict['img_fields'] = []
        input_dict['img_prefix'] = None
        return self.pipeline(input_dict)

    # ---- IoU@0.5 evaluation -------------------------------------------------
    def evaluate(self, results, logger=None, **kwargs):
        # IoU@0.5 over the full BEV grid against the visibility-filtered GT
        # (3DOD-style: low-vis vehicles are background, so predicting them is a
        # false positive). Matches the detection models' visibility>=2 protocol.
        inter = union = 0
        for index, pred in enumerate(results):
            p = (pred >= 0.5).astype(np.uint8).reshape(self.bev, self.bev)
            scene, fi, fr = self.frames[index]
            g = self._gt_mask(scene, fr)
            inter += int(((p == 1) & (g == 1)).sum())
            union += int(((p == 1) | (g == 1)).sum())
        iou = inter / max(union, 1)
        if logger:
            logger.info(f'[CARLA-SEG] vehicle IoU@0.5 = {iou:.4f}')
        print(f'[CARLA-SEG] vehicle IoU@0.5 = {iou:.4f}')
        return {'vehicle_IoU': iou}

    def _vis_mask(self, scene, fr):
        from PIL import Image
        v = np.array(Image.open(os.path.join(self.labels_dir, scene, fr['visibility'])))
        return v[::-1, ::-1].T.copy()
