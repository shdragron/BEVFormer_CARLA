"""BEVFormer-base on CARLA — BUS viewpoint.

Inherits everything from bevformer_base_carla.py (the sedan config) and only
repoints ann_file to the BUS info pkls and renames the wandb run. The scene
split (split/{train,val}.txt) is shared across all vehicles; only the source
DB (v1.0-carla_bus / v1.0-carla_bus_eval) differs, which is already baked into
the bus_infos_{train,val}.pkl produced by tools/create_carla_data.py.
"""
_base_ = ['./bevformer_base_carla.py']

vehicle = 'bus'
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
                 name=f'bevformer_base_carla_{vehicle}_bs8_workers16',
                 tags=['bevformer', 'carla', vehicle, 'temporal_off',
                       'visibility2', 'bs8', 'workers16', 'B200x2'],
                 notes=f'BEVFormer-base on CARLA jeongtae ({vehicle} viewpoint). '
                       'temporal disabled. 6 classes. visibility>=2 valid_flag. '
                       'samples_per_gpu=4 x 2 B200 = global batch 8.'),
             interval=50,
             commit=True,
             by_epoch=True,
             with_step=True,
             log_artifact=False),
    ])
