import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from typing import Any
sys.path.append(os.path.abspath('./'))
from loguru import logger
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.configuration import parse_args
from utils.utilities import Console, Ploter
from model.solver import MotionSolver
from model.dataloader import  DatasetGTA

def get_dataloader(args: Any, phase: str):
    shuffle = True if phase == 'train' or phase =='val' else False
    actions = ['walk','sit','lie','stand up']
    dataset = DatasetGTA(phase,{})

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=True)
    return dataloader

def train(args):
    """ training portal

    Args:
        args: config arguments
    """
    args.body_feat_size = args.num_betas
    args.scene_group_size = args.npoints // 256 # need change if scene model changes
    args.input_size = 3 + 6 + 63 # trans + orient + body pose
    print("sdf out", args.sdf_out)
    print("sdf out", args.sdata)
    train_dataloader= get_dataloader(args, 'train')
    val_dataloader = get_dataloader(args, 'test')

    dataloader = {
        'train': train_dataloader,
        'val': val_dataloader,
    }

    Console.log('-' * 30)
    Console.log('\n[Info]')
    Console.log("Train examples: {}".format(len(train_dataloader)))
    Console.log("Eval examples: {}".format(len(val_dataloader)))

    solver = MotionSolver(args, dataloader)
    # solver._report_model_size()
    # solver.check_data()
    # solver.visualize_prediction()
    Console.log('Start training...\n')
    solver()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    ## Reproducible
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    ## parse input arguments
    args = parse_args()

    ## set logger path
    args.log_dir = os.path.join(args.log_dir, args.stamp)
    os.makedirs(args.log_dir, exist_ok=True)
    ## set tensorboard and text logger
    logger.add(os.path.join(args.log_dir, 'runtime.log'), format='{time:YYYY-MM-DD HH:mm:ss.SSS} {level} {message}')
    writer = SummaryWriter(log_dir=args.log_dir)

    Console.setPrinter(printer=logger.info, debug=False)
    Ploter.setWriter(writer=writer)
    
    Console.log('Init logger...')
    Console.log('[************ Global Configuration ************] \n' + str(args) + '\n')

    ## set cuda
    # if args.device == 'cuda':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device("cuda:0" if args.device == 'cuda' else "cpu")
    
    train(args)
