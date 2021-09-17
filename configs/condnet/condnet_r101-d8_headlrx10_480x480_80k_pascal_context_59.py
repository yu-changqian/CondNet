_base_ = [
    '../_base_/models/condnet_r50-d8.py',
    '../_base_/datasets/pascal_context_59.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=59),
    auxiliary_head=dict(num_classes=59))
test_cfg = dict(mode='slide', crop_size=(480, 480), stride=(320, 320))
optimizer = dict(
    type='SGD',
    lr=0.004,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
