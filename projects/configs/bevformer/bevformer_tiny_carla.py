"""BEVFormer-tiny adapted for CARLA jeongtae dataset.

Diffs from bevformer_tiny.py:
  * temporal off:  queue_length=2 (min for forward_train) + scene_token=token
                   in pkl makes prev_bev_exists=False always; rotate_prev_bev=False;
                   use_can_bus=False; video_test_mode=False
  * 6 custom classes: car, truck, bus, motorcycle, bicycle, pedestrian
  * per-vehicle (viewpoint) data under data/nuscenes/ (-> carla_geobev). This is
    the SEDAN config; vehicle='sedan'. suv/bus variants: bevformer_tiny_carla_{suv,bus}.py
  * ann_file = {vehicle}_infos_train.pkl (v1.0-carla_{veh}),
               {vehicle}_infos_val.pkl   (v1.0-carla_{veh}_eval)
  * use_valid_flag=True (visibility>='2' set by converter)
  * Custom CarlaNuScenesDataset wraps eval() to accept v1.0-carla_{veh}_eval version
  * samples_per_gpu=4 x 2 GPUs = global batch 8 (matches original 8-GPU x bs1)
  * workers_per_gpu=16
  * WandbLoggerHook added

Original bevformer_tiny.py is NOT modified.

Architecture from tiny:
  - R50 backbone (not R101-DCN)
  - BEV 50x50 (not 200x200)
  - num_levels=1, encoder layers=3 (not 4, 6)
  - input image scaled to 800x450 (RandomScaleImageMultiViewImage scales=[0.5])
"""
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Coordinate frame identical to nuScenes; keep ±51.2m range.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 6 CARLA classes (others map via NuScenesDataset.NameMapping; absent in data)
class_names = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian'
]
num_classes = len(class_names)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 2  # min to satisfy forward_train; unique scene_token -> prev_bev=None always

model = dict(
    type='BEVFormer',
    use_grid_mask=False,                             # aug off (fair comparison)
    video_test_mode=False,                       # temporal off
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=False,               # temporal off
            use_shift=True,
            use_can_bus=False,                   # CAN bus off
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'CarlaNuScenesDataset'
data_root = 'data/nuscenes/'   # symlinked to /NHNHOME/.../jeongtae/carla_geobev
vehicle = 'sedan'              # ego-vehicle viewpoint; suv/bus variants override this
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # PhotoMetricDistortion removed: all augmentation off for fair comparison.
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=class_names,
                 with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,   # global batch 16 across 2 B200 (was 4). Caps at <=10:
                         # spatial cross-attn batches bs*num_cams(6) and the
                         # deformable-attn im2col_step=64 needs bs*6 <= 64 (or a
                         # multiple of 64). 8*6=48 OK; 16*6=96 fails. batch 16
                         # also matches BEVDepth for fair comparison.
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f'{vehicle}_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + f'{vehicle}_infos_val.pkl',
             pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + f'{vehicle}_infos_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,  # 2x of 2e-4 for the 4x larger batch (moderate scaling for stability)
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
evaluation = dict(interval=2, pipeline=test_pipeline)  # eval every 2 epochs

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='BEVFormer-CARLA',
                 name=f'bevformer_tiny_carla_{vehicle}_bs16_workers16',
                 tags=['bevformer', 'tiny', 'carla', vehicle, 'temporal_off',
                       'visibility2', 'bs16', 'workers16', 'B200x2'],
                 notes='BEVFormer-tiny on CARLA jeongtae. R50 backbone, '
                       'BEV 50x50, 3 encoder layers. queue_length=2 + '
                       'unique scene_token (temporal disabled). 6 classes. '
                       'visibility>=2 valid_flag. samples_per_gpu=4 x 2 B200.'),
             interval=50,
             commit=True,
             by_epoch=True,
             with_step=True,
             log_artifact=False),
    ])

checkpoint_config = dict(interval=1)
