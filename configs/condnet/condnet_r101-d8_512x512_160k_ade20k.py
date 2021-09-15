_base_ = './condnet_r50-d8_512x512_160k_ade20k.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(342, 342))
