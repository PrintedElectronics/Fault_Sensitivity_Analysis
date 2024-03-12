import sys
import os
from pathlib import Path
import pickle
import torch
import pprint
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
from utils import *
from configuration import *

if not os.path.exists('./evaluation/'):
    os.makedirs('./evaluation/')

args = parser.parse_args()
args = FormulateArgs(args)

valid_loader, datainfo = GetDataLoader(args, 'valid', path='../dataset/')
test_loader , datainfo = GetDataLoader(args, 'test',  path='../dataset/')
for x,y in valid_loader:
    X_valid, y_valid = x.to(args.DEVICE), y.to(args.DEVICE)
for x,y in test_loader:
    X_test, y_test = x.to(args.DEVICE), y.to(args.DEVICE)

if not os.path.exists(f"./evaluation/result_data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}_eTrain_{args.e_train}_eTest{args.e_test}.matrix"):

    FAULTS = [0, 1, 2, 4, 8, 16]
    N_Faults = 500
    
    results = torch.zeros([len(FAULTS), N_Faults, 6])
    
    evaluator = Evaluator(args).to(args.DEVICE)
    
    modelname = f"data_{args.DATASET:02d}_{datainfo['dataname']}_seed:{args.SEED:02d}_epsilon:{args.e_train}.model"
    model = torch.load(f'./models/{modelname}', map_location=args.DEVICE)
    model.UpdateArgs(args)
    
    model.UpdateVariation(args.N_test, args.e_test)

    SetSeed(args.SEED)
    
    for f, fault in enumerate(FAULTS):
        for faultsample in range(N_Faults):
    
            model.RemoveFault()
            model.SampleFault(fault)

            with torch.no_grad():
                result_valid = evaluator(model, X_valid, y_valid)
                valid_acc, _, valid_power, valid_area = result_valid['acc'], result_valid['std'], result_valid['power'], result_valid['area']
                result_test  = evaluator(model, X_test,  y_test)
                test_acc, _, test_power, test_area = result_test['acc'], result_test['std'], result_test['power'], result_test['area']
    
            results[f, faultsample, 0] = valid_acc
            results[f, faultsample, 1] = valid_power
            results[f, faultsample, 2] = valid_area
            results[f, faultsample, 3] = test_acc
            results[f, faultsample, 4] = test_power
            results[f, faultsample, 5] = test_area
            
                        
    torch.save(results, f"./evaluation/result_data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}_eTrain_{args.e_train}_eTest{args.e_test}.matrix")