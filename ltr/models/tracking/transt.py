import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from ltr.models.backbone import resnet

class CSANT(nn.Module):
    def __init__(self, backbone, featurefusion_network, source_mapping, target_mapping, num_classes):

        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.class_chikusei = MLP(hidden_dim, hidden_dim, 14, 3)
        self.class_ip = MLP(hidden_dim, hidden_dim, 16, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.source_mapping = source_mapping
        self.target_mapping = target_mapping

    def forward(self, search, template, name):

        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
            kind_num = template.shape[1]
            template_NT = []
        for i in range(kind_num):
            if not isinstance(template[:, i, :, :, :], NestedTensor):
                template_NT.append(nested_tensor_from_tensor(template[:, i, :, :, :]))
        feature_search, pos_search = self.backbone(search)
        for i in range(len(template_NT)):
            if i == 0:
                feature_template, pos_template = self.backbone(template_NT[i])#获得feature_template，里面包含mask[38,16,16],tensors[38,1024,16,16]，pos_template为位置编码[38,1024,16,16]
                src_template, mask_template = feature_template[-1].decompose()
                src_template = src_template.unsqueeze(0)
                mask_template = mask_template.unsqueeze(0)
                pos_template = pos_template[-1]
            else:
                feature_template_tmp, pos_template_tmp = self.backbone(template_NT[i])
                src_template_tmp, mask_template_tmp = feature_template_tmp[-1].decompose()
                src_template_tmp = src_template_tmp.unsqueeze(0)
                src_template = torch.cat([src_template, src_template_tmp], dim=0)


        src_search, mask_search= feature_search[-1].decompose()
        assert mask_search is not None
        assert mask_template is not None
        #hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])
        hs = self.featurefusion_network(src_template, mask_template, src_search,
                                        mask_search, pos_template, pos_search[-1])
        if name == "train_source":
            outputs_class = self.class_chikusei(hs)
        else:
            outputs_class = self.class_ip(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def track(self, search):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

class SetCriterion(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)



    def forward(self, outputs, targets, name, i):
        src_logits = outputs['pred_logits']
        smaple_lb = targets[0]
        target_lb = targets[1].unsqueeze(1).expand_as(smaple_lb)
        target_kind = torch.eq(smaple_lb, target_lb).nonzero()[:, 1]
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_kind.unsqueeze(1))
        losses = {'loss_ce': loss_ce}
        # 测试成功率
        max_idices = torch.argmax(src_logits, 2)
        max_idices = max_idices.squeeze(1)
        class_f_b = (max_idices == targets[1]) + 0 #bool转为int
        current_success_rate = class_f_b.sum().item() * 100/targets[0].shape[0]
        if i == 1:
            if name == "train_source":
                print(f"Source success rate: {current_success_rate}%")
            else:
                print(f"Target success rate: {current_success_rate}%")

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layer_last = nn.Linear(225, 1)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = self.layer_last(x.transpose(2, 3))
        x = x.transpose(2, 3)
        return x


@model_constructor
def transt_resnet50(settings, source_band_num, target_band_num):
    num_classes = 1
    source_mapping = resnet.Mapping(source_band_num, source_band_num)
    target_mapping = resnet.Mapping(target_band_num, source_band_num)
    backbone_net = build_backbone(settings, backbone_pretrained=False)
    featurefusion_network = build_featurefusion_network(settings)
    model = CSANT(
        backbone_net,
        featurefusion_network,
        source_mapping,
        target_mapping,
        num_classes=num_classes
    )
    device = torch.device(settings.device)
    model.to(device)
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 1}
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=1, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
