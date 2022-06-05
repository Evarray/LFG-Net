_base_= './mask_rcnn_r50_lfcp_hrfim_1x_coco.py'
model = dict(
    roi_head=dict(
        mask_head=dict(
            type='CRSB',
            mask_size=128,
            dct_vector_dim=700,
        )),
    train_cfg=dict(rcnn=dict(mask_size=128)))

