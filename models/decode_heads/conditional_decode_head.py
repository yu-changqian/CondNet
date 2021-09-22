from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, ConvModule
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from mmseg.models.losses import accuracy


class ConditionalDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 num_cond_layers=1,
                 cond_layer_with_bias=False,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=1.0),
                 loss_guidance=dict(type='DiceLoss',
                                    smooth=1e-5,
                                    loss_weight=0.2),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(ConditionalDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.num_cond_layers = num_cond_layers
        self.cond_layer_with_bias = cond_layer_with_bias
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.loss_guidance = build_loss(loss_guidance)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        weight_nums, bias_nums = [], []
        for idx_layer in range(self.num_cond_layers):
            if idx_layer == 0:
                weight_nums.append(channels * self.num_classes)
                if self.cond_layer_with_bias:
                    bias_nums.append(self.num_classes)
            else:
                weight_nums.append(self.num_classes * self.num_classes)
                if self.cond_layer_with_bias:
                    bias_nums.append(self.num_classes)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.guidance_project = self.build_project(channels,
                                                   num_classes,
                                                   groups=1,
                                                   num_convs=1,
                                                   conv_cfg=conv_cfg)
        self.filter_project = self.build_project(channels * self.num_classes,
                                                 self.num_gen_params,
                                                 groups=self.num_classes,
                                                 num_convs=1,
                                                 conv_cfg=conv_cfg)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        pass

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(input=x,
                       size=inputs[0].shape[2:],
                       mode='bilinear',
                       align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]

    def build_project(self, in_channels, channels, groups, num_convs,
                      conv_cfg):
        """Build projection layer for key/query/value/out."""
        convs = []
        if num_convs <= 0:
            convs.append(nn.Identity())
        else:
            in_channels, channels = in_channels, channels
            for i in range(num_convs):
                if i == num_convs - 1:
                    convs.append(
                        ConvModule(in_channels,
                                   channels,
                                   1,
                                   groups=groups,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=None,
                                   act_cfg=None))
                else:
                    convs.append(
                        ConvModule(in_channels,
                                   channels,
                                   1,
                                   groups=groups,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=None,
                                   act_cfg=None))
                in_channels, channels = channels, channels
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        if self.cond_layer_with_bias:
            assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        batch_size = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(
            torch.split(params, weight_nums + bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                batch_size * channels, -1, 1, 1)
            if self.cond_layer_with_bias:
                bias_splits[l] = bias_splits[l].reshape(batch_size * channels)

        return weight_splits, bias_splits

    def dynamic_head_forward(self, features, weights, biases, batch_size):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, w in enumerate(weights):
            if self.cond_layer_with_bias:
                b = biases[i]
            else:
                b = None
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=batch_size)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        output = []
        guidance_mask = self.guidance_project(feat)
        output.append(guidance_mask)
        guidance_mask = F.softmax(guidance_mask, dim=1)
        # guidance_mask = torch.sigmoid(guidance_mask)
        guidance_mask = guidance_mask.reshape(*guidance_mask.shape[:2], -1)

        key = feat
        value = feat
        key = key.reshape(*key.shape[:2], -1)
        key = key.permute(0, 2, 1).contiguous()

        cond_filters = torch.matmul(guidance_mask, key)
        cond_filters = ((value.shape[2] * value.shape[3])**-1) * cond_filters
        cond_filters = cond_filters.reshape(cond_filters.shape[0], -1, 1, 1)
        cond_filters = self.filter_project(cond_filters)
        cond_filters = cond_filters.reshape(cond_filters.shape[0], -1)
        weights, biases = self.parse_dynamic_params(cond_filters,
                                                    self.num_classes,
                                                    self.weight_nums,
                                                    self.bias_nums)

        batch_size, _, h, w = value.size()
        value = value.reshape(-1, *value.shape[2:]).unsqueeze(0)
        out = self.dynamic_head_forward(value, weights, biases,
                                        batch_size).view(
                                            batch_size, self.num_classes, h, w)

        output.insert(0, out)
        return output

    @force_fp32(apply_to=('logit', ))
    def losses(self, logits, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit, cond_logit = logits
        seg_logit = resize(input=seg_logit,
                           size=seg_label.shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(seg_logit,
                                            seg_label,
                                            weight=seg_weight,
                                            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        seg_label = seg_label.unsqueeze(1).float()
        cond_label = resize(input=seg_label,
                            size=cond_logit.shape[2:],
                            mode='nearest')

        cond_label = cond_label.squeeze(1)
        loss['loss_guidance'] = self.loss_guidance(
            cond_logit, cond_label, ignore_index=self.ignore_index)
        loss['acc_guidance'] = accuracy(cond_logit, cond_label)

        return loss
