# Training the model on small version of awareness_dataset
import torch
torch.cuda.empty_cache()
import numpy as np
import random
# import datetime
import argparse 
from os import path
import math
from tqdm import tqdm

from dataset.awareness_dataset import SituationalAwarenessDataset
# from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from model.model import STCNModel
from torch.utils.data import DataLoader

from  torch.cuda.amp import autocast
import wandb
import os
#import matplotlib.pyplot as plt

def main(args):
    """
    Initial setup
    """

    print('CUDA Device count: ', torch.cuda.device_count())

    # Parse command line arguments
    # para = HyperParameters()
    # para.parse()

    # if para['benchmark']:
    #     torch.backends.cudnn.benchmark = True

    """
    Model related
    """
    model = STCNModel(args).train()
    # For loading a checkpoint to continue training
    #model.load_model("model_saves/_checkpoint_epoch11_run12_20.pth")

    total_iter = 0

    """
    Load dataset
    """
    episode_list = sorted(os.listdir(args.raw_data), reverse=False)

    val_episodes = ["cbdr8-54" , "cbdr9-23", "cbdr6-41", "wom1-21"]
    train_episodes = list(set(episode_list) - set(val_episodes))
    # val_episodes = ["cbdr8-54"]
    # train_episodes = ["cbdr9-23", "cbdr6-41", "abd-21"]
    train_batch_size = args.batch_size

    data = []
    for ep in train_episodes:
            dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
            data.append(dataset)
            #concat_val_sample_weights += dataset.get_sample_weights()
    train_dataset = torch.utils.data.ConcatDataset(data)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(small_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.num_workers)

    val_data = []
    for ep in val_episodes:
        val_dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
        val_data.append(val_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=args.num_workers)
    

    """
    Determine current/max epoch
    """
    print('Training loader size:', len(train_loader))
    total_epoch = args.num_epochs
    current_epoch = 0
    #current_epoch = 12
    print('Number of training epochs (the last epoch might not complete): ', total_epoch)

    """
    wandb setup
    """
    wandb.init(entity='harplab-SA', project='dreyevr_stcn', name="restart_full_set_training_epochs_20_batch_size_24", config=vars(args))
    #wandb.watch(model, log='all')

    """
    Starts training
    """
    # Need this to select random bases in different workers
    # np.random.seed(np.random.randint(2**30-1) + local_rank*100)
    model.save_checkpoint(total_iter)
    for e in range(current_epoch, total_epoch): 
        print('Epoch %d/%d' % (e, total_epoch))

        total_train_loss = 0
        total_train_acc = 0

        # Train loop
        model.train()
        for data in tqdm(train_loader, desc=f"Epoch {e}", leave=False):
            with autocast():
                curr_loss, curr_acc = model.do_pass(data, e, total_iter)
            total_train_loss += curr_loss
            total_train_acc += curr_acc
            total_iter += 1
        train_loss = total_train_loss / len(train_loader)
        train_acc = total_train_acc / len(train_loader)
        wandb.log({'train_loss': train_loss, 'train_object_level_accuracy': train_acc})
        
        # validation every 2 epochs
        if e % 2 == 0:
            model.val()
            print("Validation")
            total_val_loss = 0
            total_val_acc = 0
            with torch.no_grad():
                for data_val in val_loader:
                    with autocast():
                        curr_loss, curr_acc = model.val_pass(data_val, e, total_iter)
                    total_val_loss += curr_loss
                    total_val_acc += curr_acc
            
            val_loss = total_val_loss / len(val_loader)
            val_acc = total_val_acc / len(val_loader)
            wandb.log({'val_loss': val_loss, 'val_object_level_accuracy': val_acc})
            
        # Save model checkpoint at the end of each epoch
        model.save_checkpoint(total_iter)
        
        
    # Save checkpoint at the end            
    if not args.debug:
        model.save_checkpoint(total_iter)

    
    # Vizualize the results and log to wandb
    with torch.no_grad():
        for train_data in train_loader:
            model.viz_pass(train_data, "train", total_iter)
        for val_data in val_loader:
            model.viz_pass(val_data, "val", total_iter)

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
    args.add_argument("--raw-data", type=str, default='/home/harpadmin/raw_data_corrected')
    args.add_argument("--use-rgb", action='store_true')
    args.add_argument("--instseg-channels", type=int, default=1)
    args.add_argument("--middle-andsides", action='store_true')
    args.add_argument("--secs-of-history", type=float, default=5.0)
    args.add_argument("--history-sample-rate", type=float, default=4.0)
    args.add_argument("--gaze-gaussian-sigma", type=float, default=5.0)
    args.add_argument("--gaze-fade", action='store_true')
    args.add_argument("--gaze-format", choices=['dot', 'blob'], default='dot')
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
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--num-val-episodes", type=int, default=5)
    args.add_argument("--num-epochs", type=int, default=20)
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--wandb", action='store_false')
    args.add_argument("--dont-log-images", action='store_true')
    args.add_argument("--image-save-freq", type=int, default=150)
    args.add_argument("--unfix-valset", action='store_true')
    
    #STCN Hyper params
    args.add_argument('--benchmark', action='store_true')
    args.add_argument('--amp', action='store_false')
    args.add_argument("--single_object", default=True)

    # Generic learning parameters
    args.add_argument('-i', '--iterations',default=1000, type=int)
    args.add_argument('--steps',  nargs="*", default=[150000])
    args.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)

    # Loading
    args.add_argument('--load_network', help='Path to pretrained network weight only')
    args.add_argument('--load_model', help='Path to the model file, including network, optimizer and such')

    # Logging information
    args.add_argument('--id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')
    args.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

    # Multiprocessing parameters, not set by users
    args.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')
    
    args.add_argument("--run-name", type=str, default="")
        
    args = args.parse_args()

    main(args)    
