import os
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


class LiraDataset(Dataset):
  def __init__(self, images, labels, transform=None):

    if isinstance(images, np.ndarray):
      if images.ndim == 4 and images.shape[-1] in [1, 3]:
        images = np.transpose(images, (0, 3, 1, 2))  # Convert to (N, C, H, W)
      self.images = torch.from_numpy(images).float()
    else:
      self.images = images
    
    if isinstance(labels, np.ndarray):
      self.labels = torch.from_numpy(labels).long()
    else:
      self.labels = labels

    self.transform = transform

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    image = self.images[idx]
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image)

    return image, label
  

def augmentation_fn(augmentation_type, dataset='cifar10'):
  """Applies data augmentation to images."""
  image_size = 32 if dataset in ['cifar10', 'cifar100'] else 28

  if augmentation_type == 'weak':
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(image_size)
    ])
  elif augmentation_type == 'mirror':
    return transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])
  elif augmentation_type == 'none':
    return None
  else:
    raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def get_data(dataset_name='cifar10', 
             data_dir='./data', 
             log_dir='./.experiments',
             seed=42, 
             pkeep=0.5, 
             augment='weak', 
             batch_size=64, 
             num_experiments=None, 
             expid=None, 
             only_subset=None
        ):
  """
  Load and prepare dataset for experiements.
  """
  os.makedirs(data_dir, exist_ok=True)

  train_cache = os.path.join(data_dir, f'{dataset_name}_x_train.npy')
  train_labels_cache = os.path.join(data_dir, f'{dataset_name}_y_train.npy')

  if os.path.exists(train_cache) and os.path.exists(train_labels_cache):
    inputs = np.load(train_cache)
    labels = np.load(train_labels_cache)
  else:
    if dataset_name == 'cifar10':
      dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
      inputs = dataset.data
      labels = np.array(dataset.targets)
    else:
      raise ValueError(f"Dataset {dataset_name} not supported.")
    
    #Normalize inputs to [-1, 1]
    inputs = (inputs.astype(np.float32) / 127.5) - 1.0

    np.save(train_cache, inputs)
    np.save(train_labels_cache, labels)

  nlcass = int(np.max(labels)) + 1
  dataset_size = len(inputs)

  #Subset Selection
  np.random.seed(seed)
  if num_experiments is not None:
    np.random.seed(0)
    keep = np.random.uniform(0, 1, size=(num_experiments, dataset_size))
    order = keep.argsort(0)
    keep = order < int(pkeep * num_experiments)
    keep = np.array(keep[expid], dtype=bool)
  else:
    keep = np.random.uniform(0, 1, size=dataset_size) < pkeep

  if only_subset is not None:
    keep[only_subset:] = False

  xs = inputs[keep]
  ys = labels[keep]

  train_transform = augmentation_fn(augment, dataset=dataset_name)
  train_dataset = LiraDataset(xs, ys, transform=train_transform)

  if dataset_name == 'cifar10':
    test_dataset_raw = datasets.CIFAR10(root=data_dir, train=False, download=True)
    test_images = test_dataset_raw.data
    test_labels = np.array(test_dataset_raw.targets)

  test_images = (test_images.astype(np.float32) / 127.5) - 1.0
  test_images = np.transpose(test_images, (0, 3, 1, 2))  # Convert to (N, C, H, W)
  test_dataset = LiraDataset(test_images, test_labels, transform=None)

  train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True 
                        )

  test_loader = DataLoader(test_dataset, 
                           batch_size=batch_size, 
                           shuffle=False, 
                           num_workers=4,
                           pin_memory=True
                       )  
  return train_loader, test_loader, keep, nlcass