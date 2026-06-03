"""BEVFormer-tiny on CARLA — SUV viewpoint.

Inherits everything from bevformer_tiny_carla.py (the sedan config) and only
repoints ann_file to the SUV info pkls and renames the wandb run. The scene
split (split/{train,val}.txt) is shared across all vehicles; only the source
DB (v1.0-carla_suv / v1.0-carla_suv_eval) differs, which is already baked into
the suv_infos_{train,val}.pkl produced by tools/create_carla_data.py.
"""
_base_ = ['./bevformer_tiny_carla.py']

vehicle = 'suv'
data_root = 'data/nuscenes/'

data = dict(
    train=dict(ann_file=data_root + f'{vehicle}_infos_train.pkl'),
    val=dict(ann_file=data_root + f'{vehicle}_infos_val.pkl'),
    test=dict(ann_file=data_root + f'{vehicle}_infos_val.pkl'),
)

# log_config.hooks is a list -> replaced (not deep-merged), so redefine it to
# change the wandb run name/tags for this vehicle.
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='BEVFormer-CARLA',
                 name=f'bevformer_tiny_carla_{vehicle}_bs8_workers16',
                 tags=['bevformer', 'tiny', 'carla', vehicle, 'temporal_off',
                       'visibility2', 'bs8', 'workers16', 'B200x2'],
                 notes=f'BEVFormer-tiny on CARLA jeongtae ({vehicle} viewpoint). '
                       'R50 backbone, BEV 50x50, 3 encoder layers. '
                       'temporal disabled. 6 classes. visibility>=2 valid_flag.'),
             interval=50,
             commit=True,
             by_epoch=True,
             with_step=True,
             log_artifact=False),
    ])
