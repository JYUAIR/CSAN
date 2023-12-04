from . import BaseActor
import torch
import numpy as np

class CSANActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data, name, i):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['img_target'], data['img_template'], name)
        if name == "test_target": #如果是用来测试的，直接返回网络输出结果就行
            return outputs
        #outputs = self.net(data['search_images'], data['template_images'])#encoder,deconder,和class，regression

        # generate labels
        targets = []
        label_template = data['label_template']
        label_target = data['label_target']
        #targets_origin = data['search_anno']
        #label = (label_template == label_target) + 0 # 标签，0为不同类，1为同类
        #label = label.type_as(outputs["pred_logits"])
        loss_dict = self.objective(outputs, [label_template, label_target], name, i)
        # for i in range(len(targets_origin)):
        #     h, w =data['search_images'][i][0].shape
        #     target_origin = targets_origin[i]
        #     target = {}
        #     target_origin = target_origin.reshape([1, -1])
        #     target_origin[0][0] += target_origin[0][2] / 2
        #     target_origin[0][0] /= w
        #     target_origin[0][1] += target_origin[0][3] / 2
        #     target_origin[0][1] /= h
        #     target_origin[0][2] /= w
        #     target_origin[0][3] /= h#中点宽高归一
        #     target['boxes'] = target_origin
        #     label = np.array([0])
        #     label = torch.tensor(label, device=data['search_anno'].device)
        #     target['labels'] = label
        #     targets.append(target)#中点点(x,y)，长，宽，归一化

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        # loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)#安装权重将loss加起来

        # Return training stats
        # stats = {'Loss/total': losses.item(),
        #          'Loss/ce': loss_dict['loss_ce'].item(),
        #          'Loss/bbox': loss_dict['loss_bbox'].item(),
        #          'Loss/giou': loss_dict['loss_giou'].item(),
        #          'iou': loss_dict['iou'].item()
        #          }
        if name ==  "train_source":
            stats = {'Loss_sc/total': losses.item(),
                     'Loss_sc/ce': loss_dict['loss_ce'].item()
                     }
        else:
            stats = {'Loss_tg/total': losses.item(),
                     'Loss_tg/ce': loss_dict['loss_ce'].item()
                     }
        return losses, stats
