import yaml
import os
import re
import torch
import numpy as np
import argparse
import torch.optim as optim
import multiprocessing as mp
from torch.optim.swa_utils import AveragedModel

from models import  WideResNet
from dataset import get_data

def get_logits(model, xbatch, shift=0, reflect=True, stride=1):
    if isinstance(xbatch, np.ndarray):
        xbatch = torch.from_numpy(xbatch)
    if xbatch.shape[1] != 3:  # convert NHWC → NCHW
        xbatch = xbatch.permute(0, 3, 1, 2).contiguous()

    outs = []
    batch_variants = [xbatch]
    if reflect:
        batch_variants.append(torch.flip(xbatch, dims=[3]))

    for aug in batch_variants:
        aug_pad = torch.nn.functional.pad(aug, (0, 0, shift, shift, shift, shift), mode='reflect')
        for dx in range(0, 2 * shift + 1, stride):
            for dy in range(0, 2 * shift + 1, stride):
                this_x = aug_pad[:, :, dx:dx + 32, dy:dy + 32]
                with torch.no_grad():
                    logits = model(this_x)
                outs.append(logits)

    outs = torch.stack(outs, dim=1).cpu().numpy()
    return outs

def load(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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

    model = WideResNet(depth=config['layers'], 
                       widen_factor=config['widen_factor'], 
                       num_classes=nclass,
                       dropRate=0.0
                    )
    return model, config, train_dataloader, test_dataloader

def generate_logits(args, model=None):
    xs_all = np.load(os.path.join(args.data_dir,f"{args.dataset_name}_x_train.npy"))[:args.dataset_size]
    for path in (sorted(os.listdir(os.path.join(args.log_dir)))):
        if re.search(".*experiment.*", path) is None:
            continue
        exp_config_path = os.path.join(args.log_dir, path, "config.yaml")
        
        if model is None:
            model, config, train_dataloader, test_dataloader = load(exp_config_path)
            ema_model = AveragedModel(model)
            model.eval()    

        print(f"Generating logits for {path}")
        if not os.path.exists(os.path.join(args.log_dir, path, f'best_model.pth')):
            continue
          
        if os.path.exists(os.path.join(args.log_dir, path, "logits.npy")):
            print("Skipping already generated file for experiment", path)
            continue
          
        try:
            checkpoint_path = os.path.join(args.log_dir, path, f'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
            ema_model = AveragedModel(model)
            ema_model.load_state_dict(checkpoint)
            ema_model.eval()

        except Exception as e:
            print(f"Could not load checkpoint for experiment {path}: {e}")
            continue

        n_logits = []
        for i in range(0, len(xs_all), dataset_size):
              logits = get_logits(ema_model, xs_all[i:i+dataset_size], shift=0, reflect=True, stride=1)
              n_logits.append(logits)

        if not n_logits:
              continue
          
        epoch_logits = np.concatenate(n_logits, axis=0) 
        np.save(os.path.join(args.log_dir, path, "logits.npy"), np.array(epoch_logits)[:,None,:,:])

def _softmax_logits(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits, dtype=np.float64)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs

def generate_scores(logdir, base, labels):
    if not os.path.exists(os.path.join(logdir, base)):
        print(f"Experiment directory {base} does not exist.")
        return None

    opredictions = np.load(os.path.join(logdir, base, 'logits.npy'))
    predictions = _softmax_logits(opredictions)

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT),:,:,labels[:COUNT]]

    print('mean accuracy: ',np.mean(predictions[:,0,0,:].argmax(1)==labels[:COUNT]))
    
    predictions[np.arange(COUNT),:,:,labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=3)

    logit = (np.log(y_true.mean((1))+1e-45) - np.log(y_wrong.mean((1))+1e-45))
    print("Saving scores for experiment", base)
    np.save(os.path.join(logdir, base, 'scores.npy'), logit)
    
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./.experiments', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./.data', help='Directory containing the dataset')
    parser.add_argument('--dataset_size', type=int, default=5000, help='Number of training examples to use')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='Name of the dataset (e.g., cifar10, cifar100)')
    args = parser.parse_args()
    
    model = None
    dataset_size = args.dataset_size
    generate_logits(args)
    labels = np.load(os.path.join(args.data_dir,f"{args.dataset_name}_y_train.npy"))
    
    with mp.Pool(8) as p:
        p.starmap(generate_scores, [(args.log_dir, x, labels) for x in os.listdir(args.log_dir) if 'experiment' in x])
