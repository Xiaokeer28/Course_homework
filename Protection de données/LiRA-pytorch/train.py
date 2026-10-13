import os
import time
import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter

from models import WideResNet
from dataset import get_data


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=25, delta=5, eval_steps=10, save_steps=10, log_dir='./.experiments', device='cuda'):
    since = time.time()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    best_model_path = os.path.join(log_dir, 'best_model.pth')
    best_acc = 0.0
    epoch_acc = 0.0
    best_acc_epoch = -1
    

    train_size = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    tb_dir = os.path.join(log_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            model.train()  
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if torch.isnan(loss):
                        raise ValueError(f'NaN loss detected at epoch {epoch + 1}, batch {batch_idx}')
                            
                    loss.backward()
                    optimizer.step()
                    ema_model.update_parameters(model)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            scheduler.step()

            epoch_loss = running_loss / train_size
            epoch_acc = running_corrects / train_size
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            writer.add_scalar('train/loss', epoch_loss, epoch + 1)
            writer.add_scalar('train/acc', epoch_acc, epoch + 1)
            writer.add_scalar('train/lr', current_lr, epoch + 1)

            if (epoch + 1) % eval_steps == 0 and test_dataloader is not None:
                model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        outputs = ema_model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()

                epoch_loss = running_loss / test_size
                epoch_acc = running_corrects / test_size
                print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}') 
                writer.add_scalar('val/loss', epoch_loss, epoch + 1)
                writer.add_scalar('val/acc', epoch_acc, epoch + 1)

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_epoch = epoch
                torch.save(ema_model.state_dict(), best_model_path)  
                print(f'✓ Best model updated! Accuracy: {best_acc:.4f}')

            elif delta>0 and epoch - best_acc_epoch >= delta:
                print(f'No improvement for {delta} epochs. Stopping training.')
                print(f'Best accuracy: {best_acc:.4f} at epoch {best_acc_epoch + 1}')

                checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_acc': best_acc,
                            }, checkpoint_path)
                ema_model.load_state_dict(torch.load(best_model_path, weights_only=True))
                return ema_model

            if (epoch + 1) % save_steps == 0:
                checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_acc': best_acc,
                            'best_acc_epoch': best_acc_epoch,
                            }, checkpoint_path)
                print(f'Checkpoint saved at epoch {epoch + 1}')


        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        if os.path.exists(best_model_path):
            ema_model.load_state_dict(torch.load(best_model_path, weights_only=True))
            model = ema_model
        else:
            print('No best model found, using final epoch weights')
        
        return model
    finally:
        writer.close()

def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Train a ConvNet with config and command-line overrides.")
    parser.add_argument('--config', type=str, default="config.yml", help="Path to config YAML file.")
    parser.add_argument('--dataset_name', type=str, help="Dataset name")
    parser.add_argument('--data_dir', type=str, help="Data directory")
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--pkeep', type=float, help="Keep probability")
    parser.add_argument('--augment', type=str, help="Augmentation type")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--num_experiments', type=int, help="Number of experiments")
    parser.add_argument('--expid', type=int, help="Experiment ID")
    parser.add_argument('--only_subset', type=int, help="Subset size")
    parser.add_argument('--scales', type=int, help="Number of scales")
    parser.add_argument('--filters', type=int, help="Initial number of filters")
    parser.add_argument('--filters_max', type=int, help="Maximum number of filters")
    parser.add_argument('--pooling', type=str, help="Pooling type")
    parser.add_argument('--learning_rate', type=float, help="Learning rate")
    parser.add_argument('--momentum', type=float, help="SGD momentum")
    parser.add_argument('--weight_decay', type=float, help="Weight decay")
    parser.add_argument('--step_size', type=int, help="LR scheduler step size")
    parser.add_argument('--gamma', type=float, help="LR scheduler gamma")
    parser.add_argument('--num_epochs', type=int, help="Number of epochs")
    parser.add_argument('--delta', type=int, help="Early stopping delta")
    parser.add_argument('--eval_steps', type=int, help="Evaluation steps")
    parser.add_argument('--save_steps', type=int, help="Checkpoint save steps")
    parser.add_argument('--log_dir', type=str, help="Log directory")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command-line args if provided
    for key, value in vars(args).items():
        if value is not None and key != "config":
            config[key] = value

    return config

if __name__ == '__main__':
    config = parse_args_and_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['expid'] is None and config['num_experiments'] is not None:
        raise ValueError("expid must be specified if num_experiments is set.")
    else:
        log_path = os.path.join(config['log_dir'], f"experiment_{config['expid']}_{config['num_experiments']}" if config['expid'] is not None else "experiment_full")
        config['log_dir'] = log_path
    
    if os.path.exists(os.path.join(config['log_dir'], f'checkpoint_epoch_{config["num_epochs"]}.pth')):
        print(f"Experiment already completed at {config['log_dir']}.")
        exit(0)

    os.makedirs(config['log_dir'], exist_ok=True)
    
    train_dataloader, test_dataloader, keep, nclass = get_data(
        dataset_name=config['dataset_name'],
        data_dir=config['data_dir'],
        seed=config['seed'],
        pkeep=config['pkeep'],
        augment=config['augment'],
        batch_size=config['batch_size'],
        num_experiments=config['num_experiments'],
        expid=config['expid'],
        only_subset=config['only_subset']
    )
    np.save(os.path.join(config['log_dir'],'keep.npy'), keep)

    model_ft = WideResNet(depth=config['layers'],
                          num_classes=nclass,
                          widen_factor=config['widen_factor'],
                          dropRate=0.0
                        )
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(model_ft.parameters(), 
                             lr=config['learning_rate'], 
                             momentum=config['momentum'], 
                             weight_decay=config['weight_decay']   
                        )

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, 
                                                      T_max=config['num_epochs']
                        )
    model_ft = train_model(model_ft, 
                           train_dataloader,
                           test_dataloader, 
                           criterion, 
                           optimizer_ft, 
                           exp_lr_scheduler,
                           num_epochs=config['num_epochs'], 
                           delta=config['delta'], 
                           eval_steps=config['eval_steps'], 
                           save_steps=config['save_steps'], 
                           log_dir=config['log_dir'],
                           device=device
                        )

