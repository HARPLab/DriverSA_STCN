# Training the model on small version of awareness_dataset
import torch
import torch.distributed as distributed
import numpy as np
import random
# import datetime
import argparse 
from os import path
import math
from dataset.awareness_dataset import SituationalAwarenessDataset
# from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from model.model import STCNModel
from torch.utils.data import DataLoader

def main(args):
    """
    Initial setup
    """

    print('CUDA Device count: ', torch.cuda.device_count())

    # Parse command line arguments
    para = HyperParameters()
    para.parse()

    if para['benchmark']:
        torch.backends.cudnn.benchmark = True

    """
    Model related
    """
    model = STCNModel(para).train()

    total_iter = 0

    """
    Load dataset
    """
    episodes = ["cbdr8-54" , "cbdr9-23", "cbdr6-41", "abd-21"]
    train_batch_size = args.batch_size

    data = []
    for ep in episodes:
            dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
            data.append(dataset)
            #concat_val_sample_weights += dataset.get_sample_weights()
    small_dataset = torch.utils.data.ConcatDataset(data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(small_dataset, shuffle=True)
    train_loader = DataLoader(small_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.num_workers)

    """
    Determine current/max epoch
    """
    total_epoch = math.ceil(para['iterations']/len(train_loader))
    current_epoch = total_iter // len(train_loader)
    print('Number of training epochs (the last epoch might not complete): ', total_epoch)

    """
    Starts training
    """
    # Need this to select random bases in different workers
    # np.random.seed(np.random.randint(2**30-1) + local_rank*100)
    for e in range(current_epoch, total_epoch): 
        print('Epoch %d/%d' % (e, total_epoch))
        # if para['stage']!=0 and e!=total_epoch and e>=increase_skip_epoch[0]:
        #     while e >= increase_skip_epoch[0]:
        #         cur_skip = skip_values[0]
        #         skip_values = skip_values[1:]
        #         increase_skip_epoch = increase_skip_epoch[1:]
        #     print('Increasing skip to: ', cur_skip)
        #     train_sampler, train_loader = renew_loader(cur_skip)

        # Crucial for randomness! 
        train_sampler.set_epoch(e)

        # Train loop
        model.train()
        for data in train_loader:
            model.do_pass(data, total_iter)
            total_iter += 1

            if total_iter >= para['iterations']:
                break
    # finally:
    #     if not para['debug'] and model.logger is not None and total_iter>5000:
    #         model.save(total_iter)
    #     # Clean up
    #     distributed.destroy_process_group()





if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # model params
    args.add_argument("--architecture", choices=['fpn', 'unet', 'deeplabv3'], default='fpn')
    args.add_argument("--encoder", choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2', 'efficientnet-b0'], default='mobilenet_v2')
    args.add_argument("--encoder-weights", choices=['imagenet', 'swsl', 'ssl', 'instagram', None], default=None)
    # args.add_argument("--classes", type=str, default='car')
    # new dice loss does activation in the loss function
    args.add_argument("--activation", choices=[None], default=None)    
    args.add_argument("--seg-mode", choices=['binary', 'multiclass', 'multilabel'], default='multiclass')

    # data set config params
    args.add_argument("--sensor-config-file", type=str, default='sensor_config.ini')
    args.add_argument("--raw-data", type=str, default='/media/storage/raw_data_corrected')
    args.add_argument("--use-rgb", action='store_true')
    args.add_argument("--instseg-channels", type=int, default=1)
    args.add_argument("--middle-andsides", action='store_true')
    args.add_argument("--secs-of-history", type=float, default=5.0)
    args.add_argument("--history-sample-rate", type=float, default=4.0)
    args.add_argument("--gaze-gaussian-sigma", type=float, default=5.0)
    args.add_argument("--gaze-fade", action='store_true')
    args.add_argument("--gaze-format", choices=['dot', 'blob'], default='blob')
    args.add_argument("--lr-decay-epochstep", type=int, default=10)
    args.add_argument("--lr-decay-factor", type=int, default=10)
    args.add_argument("--sample-clicks", choices=['post_click', 'pre_excl', 'both', ''], 
                      default='', help="Empty string -> sample everything")
    args.add_argument("--ignore-oldclicks", action='store_true')
    args.add_argument("--weighted-unaware-sampling", action='store_true', help="equally sample images with atleast one unaware obj and images with no unaware obj")
    args.add_argument("--pre-clicks-excl-time", type=float, default=1.0, help="seconds before click to exclude for reaction time")
    args.add_argument("--unaware-classwt", type=float, default=1.0)
    args.add_argument("--bg-classwt", type=float, default=1e-5)
    args.add_argument("--aware-threshold", type=float, default=0.5)
    args.add_argument("--unaware-threshold", type=float, default=0.5)


    # training params
    args.add_argument("--device", type=str, default='cuda')
    args.add_argument("--random-seed", type=int, default=999)
    args.add_argument("--num-workers", type=int, default=12)
    args.add_argument("--batch-size", type=int, default=16)
    args.add_argument("--num-val-episodes", type=int, default=5)
    args.add_argument("--num-epochs", type=int, default=40)
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--wandb", action='store_false')
    args.add_argument("--dont-log-images", action='store_true')
    args.add_argument("--image-save-freq", type=int, default=150)
    args.add_argument("--unfix-valset", action='store_true')
    
    args.add_argument("--run-name", type=str, default="")    
    args = args.parse_args()

    main(args)    