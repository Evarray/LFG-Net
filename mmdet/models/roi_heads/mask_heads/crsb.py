import numpy as np
import torch
import torch_dct
from torch.nn import functional as F
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer, kaiming_init, caffe2_xavier_init
from mmcv.runner import auto_fp16, force_fp32
from mmcv.ops import Conv2d, Linear
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import mask_target
from .fcn_mask_head import _do_paste_mask
from .fcn_mask_head import BYTES_PER_FLOAT, GPU_MEM_LIMIT

@HEADS.register_module()
class CRSB(nn.Module):
    """Compress Recovery Segmentation Branch used in `LFG-Net`_.

    Args:
        mask_size (int): size of the recovered mask in mask prediction.
        fc_out_dim (int): compressed vector dimension of fc layer.
        dct_vector_dim (int): compressed vector dimension of zig-zag operation in mask prediction.
        mask_loss_para (float): weight for the loss function.
    """

    def __init__(self,
                 mask_size=128,
                 fc_out_dim=1024,
                 dct_vector_dim=300,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=1,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 mask_loss_para=0.007,
                 loss_mask=dict(
                     type='L1Loss'),
                 ):
        super(CRSB, self).__init__()
        self.dct_vector_dim = dct_vector_dim
        self.dct_encoding = DctMaskEncoding(vec_dim=dct_vector_dim, mask_size=mask_size)

        self.num_convs = num_convs
        self.fc_out_dim = fc_out_dim
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.mask_loss_para = mask_loss_para

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding))

        self.predictor_fc1 = nn.Linear(self.conv_out_channels*self.roi_feat_size**2, self.fc_out_dim)
        self.predictor_fc2 = nn.Linear(self.fc_out_dim, self.fc_out_dim)
        self.predictor_fc3 = nn.Linear(self.fc_out_dim, dct_vector_dim)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in [self.predictor_fc1, self.predictor_fc2]:
            caffe2_xavier_init(fc)
        nn.init.normal_(self.predictor_fc3.weight, std=0.001)
        nn.init.constant_(self.predictor_fc3.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.flatten(1)
        x = F.relu(self.predictor_fc1(x))
        x = F.relu(self.predictor_fc2(x))
        x = self.predictor_fc3(x)
        return x

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, pred, target, label, reduction='mean', avg_factor=None):
        assert reduction == 'mean' and avg_factor is None
        loss = dict()
        vector_mask_targets = []

        for mask_target in target:
            encoded_mask = self.dct_encoding.encode(mask_target)
            vector_mask_targets.append(encoded_mask)

        if len(vector_mask_targets) == 0:
            return vector_mask_targets.sum() * 0
        vector_mask_targets = torch.cat(vector_mask_targets, dim=0)
        vector_mask_targets = vector_mask_targets.to(dtype=torch.float32)
        num_instances = vector_mask_targets.size()[0]
        if pred.size(0) == 0:
            loss_mask = pred.sum() * 0
        else:
            loss_mask = F.l1_loss(pred, vector_mask_targets, reduction="none")
            loss_mask = self.mask_loss_para * loss_mask / num_instances
            loss_mask = torch.sum(loss_mask)

        loss['loss_mask'] = loss_mask
        return loss

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        pred_mask_rc = []
        mask_pred = mask_pred.split(1, dim=0)
        for mask_p in mask_pred:
            pred_mask_rc.append(self.dct_encoding.decode(mask_p.detach()))
        pred_mask_rc = torch.cat(pred_mask_rc, dim=0)
        mask_pred = pred_mask_rc[:, None, :, :]
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].cpu().numpy())
        return cls_segms

class DctMaskEncoding(object):
    """
    Apply DCT to encode the binary mask, and use the encoded vector as mask representation in instance segmentation.
    """
    def __init__(self, vec_dim, mask_size=128):
        """
        vec_dim: the dimension of the encoded vector, int
        mask_size: the resolution of the initial binary mask representaiton.
        """
        self.vec_dim = vec_dim
        self.mask_size = mask_size
        assert vec_dim <= mask_size*mask_size
        self.dct_vector_coords = self.get_dct_vector_coords(r=mask_size)

    def encode(self, masks, dim=None):
        """
        Encode the mask to vector of vec_dim or specific dimention.
        """
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
        masks = masks.view([-1, self.mask_size, self.mask_size]).to(dtype=float)  # [N, H, W]
        dct_all = torch_dct.dct_2d(masks, norm='ortho')
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_all[:, xs, ys]  # reshape as vector
        return dct_vectors  # [N, D]

    def decode(self, dct_vectors, dim=None):
        """
        intput: dct_vector numpy [N,dct_dim]
        output: mask_rc mask reconstructed [N, mask_size, mask_size]
        """
        device = dct_vectors.device
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
            dct_vectors = dct_vectors[:, :dim]

        N = dct_vectors.shape[0]
        dct_trans = torch.zeros([N, self.mask_size, self.mask_size], dtype=dct_vectors.dtype).to(device)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_trans[:, xs, ys] = dct_vectors
        mask_rc = torch_dct.idct_2d(dct_trans, norm='ortho')
        return mask_rc

    def get_dct_vector_coords(self, r=128):
        """
        Get the coordinates with zigzag order.
        """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i-j, j) for j in range(i+1)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i+1)]
                dct_index.extend(index)
        for i in range(r, 2*r-1):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i-r+1, r)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i-r+1, r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs

