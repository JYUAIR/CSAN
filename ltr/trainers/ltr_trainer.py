import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
import torch
import time
from pytracking import TensorDict
import csv


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, train_loader, test_loader, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, train_loader, test_loader, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader, epoch):
        """Do a cycle of training or validation."""
        use_source = False
        self.actor.train(loader[0].training)
        torch.set_grad_enabled(loader[0].training)
        source_in_epoch = -1
        target_in_epoch = -1
        self._init_timing()
        total_time = 0
        for i, data0 in enumerate(loader[0], 1):
            data = TensorDict({'img_template': data0[0], 'img_target': data0[1],
                              'label_template': data0[2], 'label_target': data0[3]})
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            starttime = time.time()
            if use_source == True and epoch > source_in_epoch: #先计算一波目标域后，再把源域加进来
                loss_source, stats_source = self.actor(data, loader[0].name, i)
            #target数据前向传播
            data_target_iter = iter(loader[1])
            data_target = next(data_target_iter)
            data_target = TensorDict({'img_template': data_target[0], 'img_target': data_target[1],
                               'label_template': data_target[2], 'label_target': data_target[3]})
            if self.move_data_to_gpu:
                data_target = data_target.to(self.device)
            data_target['epoch'] = self.epoch
            data_target['settings'] = self.settings
            if epoch > target_in_epoch:
                loss_target, stats_target = self.actor(data_target, loader[1].name, i)
            if use_source == True and epoch > source_in_epoch and epoch > target_in_epoch:
                loss = loss_target + loss_source
            elif use_source == True and epoch > source_in_epoch:
                loss = loss_source
            else:
                loss = loss_target
            #loss = loss_target
            #stats_source.update(stats_target)
            #stats = stats_source
            # backward pass and update weights
            if loader[0].training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            endtime = time.time()
            time_use = endtime - starttime
            total_time = total_time + time_use

            # update statistics
            if use_source == True and epoch > source_in_epoch:
                batch_size = data['img_target'].shape[loader[0].stack_dim]
                self._update_stats(stats_source, batch_size, loader[0])
                self._print_stats(i, loader[0], batch_size)
            if epoch > target_in_epoch:
                batch_size = data['img_target'].shape[loader[1].stack_dim]
                self._update_stats(stats_target, batch_size, loader[1])
                self._print_stats(i, loader[1], batch_size)
        print(f"One epoch take {total_time} second")
            # print statistics


    def train_epoch(self):
        """Do one epoch for each loader."""
        # for loader in self.loaders:
        #     if self.epoch % loader.epoch_interval == 0:
        #         self.cycle_dataset(loader)
        if self.epoch % self.loaders[0].epoch_interval == 0:
            self.cycle_dataset(self.loaders, self.epoch)

        # self._stats_new_epoch()
        # self._write_tensorboard()

    def test_epoch(self):
        train_loader = self.train_loader
        test_loader = self.test_loader
        self.actor.train(False)
        torch.set_grad_enabled(False)
        correct_num = 0
        wrong_num = 0
        print("Testing..........")
        testing_result = []
        idxy_list = []
        correct_num = 0
        total_num = 0
        total_time = 0
        for i, data0 in enumerate(test_loader, 1):#拿到一个测试样本
            test_data_num = data0[1].shape[0]
            for j, data in enumerate(train_loader, 1):
                data1 = TensorDict(
                    {'img_template': data[0], 'img_target': data0[0],
                     'label_template': data[2], 'label_target': data0[1]})
                if self.move_data_to_gpu:
                    data1 = data1.to(self.device)
                starttime = time.time()
                outputs = self.actor(data1, test_loader.name, i)
                endtime = time.time()
                time_cost = endtime - starttime
                total_time = total_time + time_cost

                outputs = torch.softmax(outputs["pred_logits"], 2)  # 进行softmax归一化
                outputs = outputs.squeeze(1)
                target_label = data1["label_target"]
                pred_kind = outputs.argmax(dim=1)
                correct_num_batch = (pred_kind == target_label).sum()
                correct_num = correct_num + correct_num_batch
                #total_num_batch = len(pred_kind)
                total_num = total_num + test_data_num
                testing_result.append(pred_kind.tolist())
                testing_result.append(target_label.tolist())
                testing_result.append("-----------------------------------")
                idxy_list.append(data0[2].tolist())
                idxy_list.append(data0[3].tolist())
        print(f"The testing time cost {total_time} second.")
        success_rate = correct_num.item() * 100 / total_num
        print(f"---testing success rate is {success_rate}%----")
        with open(f"/home/hye/hyper/hyter_transt_templates/testing_result/class_result_{self.epoch}.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(testing_result)
        with open(f"/home/hye/hyper/hyter_transt_templates/testing_result/idxy_{self.epoch}.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(idxy_list)
        #     for k in range(test_data_num):#合成测试样本特征
        #         if k == 0:
        #             test_sample_img = torch.cat([data0[0][k].unsqueeze(0)] * train_loader.batch_size, dim=0)
        #         else:
        #             tmp = torch.cat([data0[0][k].unsqueeze(0)] * train_loader.batch_size, dim=0)
        #             test_sample_img = torch.cat([test_sample_img, tmp], dim=0)
        #     #test_sample_img = torch.cat([data0[0]]*train_loader.batch_size, dim=0)#将待识别的样本重复，做成与train_loader一样大小
        #     #test_sample_label = torch.cat([data0[1]]*train_loader.batch_size, dim=0)
        #     for k in range(test_data_num):#合成测试样本标签
        #         if k == 0:
        #             test_sample_label = torch.cat([data0[1].unsqueeze(1)[k]] * train_loader.batch_size)
        #         else:
        #             tmp = torch.cat([data0[1].unsqueeze(1)[k]] * train_loader.batch_size)
        #             test_sample_label = torch.cat([test_sample_label, tmp])
        #
        #     data_test = TensorDict({'img_template': test_sample_img, 'label_template': test_sample_label})
        #     if self.move_data_to_gpu:
        #         data_test = data_test.to(self.device)
        #
        #     data_test['epoch'] = self.epoch
        #     data_test['settings'] = self.settings
        #
        #     #拿到目标域training数据
        #     for j, data in enumerate(train_loader, 1):
        #         #合成模版特征
        #         template_fea = torch.cat([data[0]]*test_data_num, dim=0)
        #         template_lbl = torch.cat([data[2]]*test_data_num, dim=0)
        #         data_template = TensorDict({'img_template': template_fea, 'label_template': template_lbl})
        #         if self.move_data_to_gpu:
        #             data_template = data_template.to(self.device)
        #         data_template['epoch'] = self.epoch
        #         data_template['settings'] = self.settings
        #         data1 = TensorDict({'img_template': data_template["img_template"], 'img_target': data_test["img_template"],
        #                            'label_template': data_template["label_template"], 'label_target': data_test["label_template"]})
        #         outputs = self.actor(data1, test_loader.name, i)
        #         outputs = torch.softmax(outputs["pred_logits"], 2) #进行softmax归一化
        #         outputs = outputs.squeeze(1)
        #         label_template = data1["label_template"]
        #         class_list = label_template.unique()
        #         each_class_num = class_list.__len__()
        #
        #         for sample_idx in range(test_data_num):
        #             #计算一个测试样本属于哪一类
        #             max_score = -10000
        #             for k in range(each_class_num):
        #                 class_index = torch.where(label_template[sample_idx*train_loader.batch_size:(sample_idx+1)*train_loader.batch_size] == k)
        #                 class_index = class_index[0] + sample_idx * train_loader.batch_size
        #                 score = (outputs[class_index][:, 1] - outputs[class_index][:, 0]).sum()
        #                 if score > max_score:
        #                     max_score = score
        #                     class_result = k
        #             if class_result == data0[1][sample_idx]:
        #                 correct_num += 1
        #             else:
        #                 wrong_num += 1
        #                 testing_result.append([class_result, data0[1][sample_idx].item()])
        #
        # success_rate = 100 * correct_num / (wrong_num + correct_num)
        # print(f"---testing success rate is {success_rate}%----")
        # with open(f"/home/hye/hyper/hyper_transt/testing_result/class_result_{self.epoch}.csv", "w") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(testing_result)
    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)