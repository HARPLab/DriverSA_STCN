from vos_dataset import VOSDataset
from os import path
import argparse
from awareness_dataset import SituationalAwarenessDataset

davis_root = '/home/srkhuran-local/DAVIS/2016'

vos_data = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'), 
                        path.join(davis_root, 'Annotations', '480p'), 5, is_bl=False)

item = vos_data.__getitem__(20)

args = argparse.ArgumentParser()
# data set config params
args.add_argument("--seg-mode", choices=['binary', 'multiclass', 'multilabel'], default='multiclass')
args.add_argument("--sensor-config-file", type=str, default='dataset/sensor_config.ini')
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
   
args = args.parse_args()
# Lets look at data we have

dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, "cbdr10-23", args)

item = dataset.__getitem__(20)