_base_ = './atss_r50_bfp_gn_soft_nms_2x_sku110k.py'
model = dict(
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(type='Res2Net', depth=101, scales=4, base_width=26))
