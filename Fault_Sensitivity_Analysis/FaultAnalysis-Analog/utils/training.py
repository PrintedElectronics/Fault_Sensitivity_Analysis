import time
import math
from .checkpoint import *
from .evaluation import *

def train_pnn_progressive(nn, train_loader, valid_loader, lossfunction, optimizer, args, logger, UUID='default'):
    start_training_time = time.time()
    
    evaluator = Evaluator(args)
    
    best_valid_loss = math.inf
    current_lr = args.LR
    patience_lr = 0
    
    lr_update = False
    early_stop = False
    
    if load_checkpoint(UUID, args.temppath):
        current_epoch, nn, optimizer, best_valid_loss = load_checkpoint(UUID, args.temppath)
        for g in optimizer.param_groups:
            current_lr = g['lr']
            g['params'] = nn.GetParam()
        logger.info(f'Restart previous training from {current_epoch} epoch with lr: {current_lr}.')
        print(f'Restart previous training from {current_epoch} epoch with lr: {current_lr}.')
    else:
        current_epoch = 0

        
    for epoch in range(current_epoch, args.EPOCH):
        start_epoch_time = time.time()
        
        msg = ''
        
        for x_train, y_train in train_loader:
            msg += f'{current_lr}'
            msg += f'hyperparameters in printed neural network for training :\nepoch : {epoch:-6d} |\n'
            
            L_train = lossfunction(nn(x_train), y_train)
            train_result = evaluator(nn, x_train, y_train)
            train_acc, train_std, train_power, train_area = train_result['acc'], train_result['std'], train_result['power'], train_result['area']
            optimizer.zero_grad()
            L_train.backward()
            optimizer.step()
            
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                msg += f'hyperparameters in printed neural network for validation :\nepoch : {epoch:-6d} |\n'
                
                L_valid = lossfunction(nn(x_valid), y_valid)
                valid_result = evaluator(nn, x_valid, y_valid)
                valid_acc, valid_std, valid_power, valid_area = valid_result['acc'], valid_result['std'], valid_result['power'], valid_result['area']
        
        logger.debug(msg)
        
        if args.recording:
            record_checkpoint(epoch, nn, L_train, L_valid, UUID, args.recordpath)
            
        if L_valid.item() < best_valid_loss:
            best_valid_loss = L_valid.item()
            save_checkpoint(epoch, nn, optimizer, best_valid_loss, UUID, args.temppath)
            patience_lr = 0
        else:
            patience_lr += 1

        if patience_lr > args.LR_PATIENCE:
            print('lr update')
            lr_update = True
        
        if lr_update:
            lr_update = False
            patience_lr = 0
            _, nn, _,_ = load_checkpoint(UUID, args.temppath)
            logger.info('load best network to warm start training with lower lr.')
            for g in optimizer.param_groups:
                # g['params'] = [p for p in nn.parameters()]
                g['params'] = nn.GetParam()
                g['lr'] = g['lr'] * args.LR_DECAY
                current_lr = g['lr']
            logger.info(f'lr update to {current_lr}.')

        if current_lr < args.LR_MIN:
            early_stop = True
            print('early stop.')
            logger.info('Early stop.')
            break
        
        end_epoch_time = time.time()
        end_training_time = time.time()
        if (end_training_time - start_training_time) >= args.TIMELIMITATION*60*60:
            print('Time limination reached.')
            logger.warning('Time limination reached.')
            break

        if not epoch % args.report_freq:
            print(f'| Epoch: {epoch:-6d} | Train loss: {L_train.item():.4e} | Valid loss: {L_valid.item():.4e} | Train acc: {train_acc:.4f} ± {train_std:.4f} | Valid acc: {valid_acc:.4f} ± {valid_std:.4f} | patience: {patience_lr:-3d} | lr: {current_lr} | Epoch time: {end_epoch_time-start_epoch_time:.1f} |')
            logger.info(f'| Epoch: {epoch:-6d} | Train loss: {L_train.item():.4e} | Valid loss: {L_valid.item():.4e} | Train acc: {train_acc:.4f} ± {train_std:.4f} | Valid acc: {valid_acc:.4f} ± {valid_std:.4f} | patience: {patience_lr:-3d} | lr: {current_lr} | Epoch time: {end_epoch_time-start_epoch_time:.1f} |')
        
    _, resulted_nn, _,_ = load_checkpoint(UUID, args.temppath)
    
    if early_stop:
        os.remove(f'{args.temppath}/{UUID}.ckp')

    # return resulted_nn, early_stop
    return resulted_nn, early_stop




def train_pnn(nn, train_loader, valid_loader, lossfunction, optimizer, args, logger, UUID='default'):
    start_training_time = time.time()
    
    evaluator = Evaluator(args)
    
    best_valid_loss = math.inf
    patience = 0
    
    early_stop = False
    
    if load_checkpoint(UUID, args.temppath):
        current_epoch, nn, optimizer, best_valid_loss = load_checkpoint(UUID, args.temppath)
        logger.info(f'Restart previous training from {current_epoch} epoch')
        print(f'Restart previous training from {current_epoch} epoch')
    else:
        current_epoch = 0
        
    for epoch in range(current_epoch, args.EPOCH):
        start_epoch_time = time.time()
        
        msg = ''
        
        for x_train, y_train in train_loader:
            msg += f'hyperparameters in printed neural network for training :\nepoch : {epoch:-6d} |\n'
            
            L_train = lossfunction(nn, x_train, y_train)
            train_result = evaluator(nn, x_train, y_train)
            train_acc, train_std, train_power, train_area = train_result['acc'], train_result['std'], train_result['power'], train_result['area']
            optimizer.zero_grad()
            L_train.backward()
            optimizer.step()

        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                msg += f'hyperparameters in printed neural network for validation :\nepoch : {epoch:-6d} |\n'
                
                L_valid = lossfunction(nn, x_valid, y_valid)
                valid_result = evaluator(nn, x_valid, y_valid)
                valid_acc, valid_std, valid_power, valid_area = valid_result['acc'], valid_result['std'], valid_result['power'], valid_result['area']
        
        logger.debug(msg)
        
        if args.recording:
            record_checkpoint(epoch, nn, L_train, L_valid, UUID, args.recordpath)
            
        if L_valid.item() < best_valid_loss:
            best_valid_loss = L_valid.item()
            save_checkpoint(epoch, nn, optimizer, best_valid_loss, UUID, args.temppath)
            patience = 0
        else:
            patience += 1

        if patience > args.PATIENCE:
            print('Early stop.')
            logger.info('Early stop.')
            early_stop = True
            break
        
        end_epoch_time = time.time()
        end_training_time = time.time()
        if (end_training_time - start_training_time) >= args.TIMELIMITATION*60*60:
            print('Time limination reached.')
            logger.warning('Time limination reached.')
            break
        
        if not epoch % args.report_freq:
            print(f'| Epoch: {epoch:-6d} | Train loss: {L_train.item():.4e} | Valid loss: {L_valid.item():.4e} | Train acc: {train_acc:.4f} ± {train_std:.4f} | Valid acc: {valid_acc:.4f} ± {valid_std:.4f} | patience: {patience_lr:-3d} | lr: {current_lr} | Epoch time: {end_epoch_time-start_epoch_time:.1f} |')
            logger.info(f'| Epoch: {epoch:-6d} | Train loss: {L_train.item():.4e} | Valid loss: {L_valid.item():.4e} | Train acc: {train_acc:.4f} ± {train_std:.4f} | Valid acc: {valid_acc:.4f} ± {valid_std:.4f} | patience: {patience_lr:-3d} | lr: {current_lr} | Epoch time: {end_epoch_time-start_epoch_time:.1f} |')
        
    _, resulted_nn, _,_ = load_checkpoint(UUID, args.temppath)
    
    if early_stop:
        os.remove(f'{args.temppath}/{UUID}.ckp')

    return resulted_nn, early_stop

