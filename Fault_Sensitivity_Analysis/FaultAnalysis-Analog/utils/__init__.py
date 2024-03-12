import torch
import numpy as np
import random
import os
from .training import *
from .evaluation import *
from .logger import *
from .Loader import *

def CheckDevice(args):
    if args.DEVICE == 'cpu':
        return torch.device('cpu')
    elif args.DEVICE == 'gpu':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            raise RuntimeError('No GPU avaiable!')
    
def SetSeed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)                  
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def MakeFolder(args):
    if args.projectname:
        args.temppath = f'./{args.projectname}/temp/'
        args.logfilepath = f'./{args.projectname}/log/'
        args.recordpath = f'./{args.projectname}/record/'
        args.savepath = f'./{args.projectname}/models/'
    
    if not os.path.exists(args.temppath):
        os.makedirs(args.temppath)
        
    if not os.path.exists(args.logfilepath):
        os.makedirs(args.logfilepath)

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    
    if args.recording:
        if not os.path.exists(args.recordpath):
            os.makedirs(args.recordpath)

def FormulateArgs(args):
    args.DEVICE = CheckDevice(args)
    if args.e_train == 0. and args.dropout == 0.:
        args.N_train = 1
    if args.e_test == 0. and args.dropout == 0.:
        args.N_test = 1
    if args.e_fault == 0:
        args.N_fault = 1
    if args.MODE=='nominal':
        args.M_train = 1
        args.K_train = 1
    if args.InputNoise==0.:
        args.R_train = 1
    return args