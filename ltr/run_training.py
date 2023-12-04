import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn
#import auxil


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
import ltr.admin.settings as ws_settings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run_training(train_module, train_name, args_source, args, cudnn_benchmark=True):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings, args_source, args)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--dataset', default='PU_EMAP', type=str,
                        help='dataset (options: SV_EMAP, PU_EMAP, IP9_EMAP, IP, PU, SV, KSC)')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--patchsize', dest='patchsize', default=9, type=int, help='spatial patch size')
    parser.add_argument('--rand_state', default=1228, type=int, help='(None,123) Random seed')
    parser.add_argument('--tr_percent', default=5, type=float, help='(100 or 0.05) Samples of train set')
    parser.add_argument('--use_val', default=False, type=bool, help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='(0.05) samples of val set')
    parser.add_argument('--use_test', default=True, type=bool, help='Use test set')
    parser.add_argument('--tr_bsize', default=128, type=int, help='(400) Important! Mini-batch train size')
    parser.add_argument('--te_bsize', default=128, type=int, help='(1000) Mini-batch test size')
    parser.add_argument('--is_source', default=False, type=bool, help='Is the data from the source domain')
    args = parser.parse_args()
    args_source = parser.parse_args() #源域的参数
    args_source.dataset = 'Chikusei_EMAP' #源域的样本
    args_source.tr_percent = 1000
    args_source.tr_bsize = 32
    args_source.use_test = False
    args_source.use_val = False
    args_source.is_source = True


    run_training(args.train_module, args.train_name, args_source, args, args.cudnn_benchmark)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
