_base_ = './atss_r2_101_bfp_gn_soft_nms_2x_sku110k.py'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8
)
optimizer = dict(
    lr=0.0025
)
