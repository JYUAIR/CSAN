class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/hye/hyper/hyper_templates_git/ltr/workspace'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        # self.lasot_dir = '/opt/data/private/houyueen/mydata/pytracking/pytracking/ltr/dataset/LaSOT/LaSOT'
        # #self.got10k_dir = '/opt/data/private/houyueen/mydata/pytracking/pytracking/ltr/dataset/GOT-10k/full_data/train_data'
        # self.got10k_dir = '/home/hye/hyper/data/GOT-10k'
        # self.trackingnet_dir = '/opt/data/private/houyueen/mydata/pytracking/pytracking/ltr/dataset/TrackingNet/'
        # self.coco_dir = '/opt/data/private/houyueen/mydata/pytracking/pytracking/ltr/dataset/COCO2014'
        # self.lvis_dir = ''
        # self.sbd_dir = ''
        #
        #
        #
        # self.imagenet_dir = ''
        # self.imagenetdet_dir = ''
        # self.ecssd_dir = ''
        # self.hkuis_dir = ''
        # self.msra10k_dir = ''
        # self.davis_dir = ''
        # self.youtubevos_dir = ''
