_base_ = [
    # 基本 runtime 設定
    'mmrotate/configs/_base_/default_runtime.py',
    # 模型結構
    'mmrotate/configs/_base_/models/rotated_retinanet_r50_fpn.py',
    # schedule (學習率/訓練週期設定)
    'mmrotate/configs/_base_/schedules/schedule_1x.py'
]

angle_version = 'le90'  # 可能是 'le90'、'oc' 或 'le135' 等，要與標註格式對應
num_classes = 1  # 只有一個類別：Fracture

model = dict(
    bbox_head=dict(
        num_classes=num_classes,
        # anchor_generator等部分可能需要依資料大小調整
        angle_version=angle_version,
    )
)

# 資料集設定
train_json = 'train.json'
test_json = 'test.json'
img_prefix_train = 'train/frac_img'
img_prefix_test = 'test/frac_img'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),  # 注意這裡要把 rotated box 也讀進來
    # 其他資料增強 ...
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='PackRotatedInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackRotatedInputs', meta_keys=('img_id', 'img_path', 'ori_shape',
                                              'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='DOTARDataset',  # 或 RotatedCocoDataset，視實際情況
        data_root='.',
        ann_file=train_json,
        data_prefix=dict(img_path=img_prefix_train),
        pipeline=train_pipeline,
        # 若使用 RotatedCocoDataset，則要寫成 type='RotatedCocoDataset'
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='DOTARDataset',  # 或 RotatedCocoDataset
        data_root='.',
        ann_file=test_json,
        data_prefix=dict(img_path=img_prefix_test),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='DOTAMetric',  # 若使用 RotatedCocoDataset 則會用 CocoMetric
    metric='mAP'
)
test_evaluator = val_evaluator

# 訓練 epoch 數
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()
