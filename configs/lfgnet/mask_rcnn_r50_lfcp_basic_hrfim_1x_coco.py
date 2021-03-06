_base_ = './mask_rcnn_r50_lfcp_basic_1x_coco.py'
model = dict(
    roi_head=dict(
        type='HRFInteractionRoIHead',
        bbox_head=dict(
            type='IntBBoxHead',
            num_shared_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss',
                           loss_weight=1.0)),
        feat_interaction_head=dict(
            type='HRFInteractionHead',
            in_channels=1024,
            out_conv_channels=256,
            roi_feat_size=7,
            scale_factor=2)
    ))
