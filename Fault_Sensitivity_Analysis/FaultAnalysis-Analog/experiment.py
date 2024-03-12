import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from configuration import *
import torch
import pprint
import pNN
from utils import *

args = parser.parse_args()
args = FormulateArgs(args)

print(f'Training network on device: {args.DEVICE}.')
MakeFolder(args)

train_loader, datainfo = GetDataLoader(args, 'train')
valid_loader, datainfo = GetDataLoader(args, 'valid')
test_loader, datainfo = GetDataLoader(args, 'test')
pprint.pprint(datainfo)

SetSeed(args.SEED)

setup = f"data_{args.DATASET:02d}_{datainfo['dataname']}_seed_{args.SEED:02d}_epsilon_{args.e_train}_dropout_{args.dropout}.model"
print(f'Training setup: {setup}.')

msglogger = GetMessageLogger(args, setup)
msglogger.info(f'Training network on device: {args.DEVICE}.')
msglogger.info(f'Training setup: {setup}.')
msglogger.info(datainfo)

if os.path.isfile(f'{args.savepath}/{setup}'):
    print(f'{setup} exists, skip this training.')
    msglogger.info('Training was already finished.')
else:
    topology = [datainfo['N_feature']] + args.hidden + [datainfo['N_class']]
    msglogger.info(f'Topology of the network: {topology}.')

    pnn = pNN.pNN(topology, args).to(args.DEVICE)

    lossfunction = pNN.pNNLoss(args).to(args.DEVICE)
    optimizer = torch.optim.Adam(pnn.GetParam(), lr=args.LR)

    if args.PROGRESSIVE:
        pnn, best = train_pnn_progressive(pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    else:
        pnn, best = train_pnn(pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)

    if best:
        if not os.path.exists(f'{args.savepath}/'):
            os.makedirs(f'{args.savepath}/')
        torch.save(pnn, f'{args.savepath}/{setup}')
        msglogger.info('Training if finished.')
    else:
        msglogger.warning('Time out, further training is necessary.')

CloseLogger(msglogger)