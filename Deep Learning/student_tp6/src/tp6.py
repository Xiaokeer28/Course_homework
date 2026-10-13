# %%
import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click


from sklearn.datasets import fetch_openml

# Changer le DATA_PATH
DATA_PATH = "/tmp/mnist"


# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
TEST_RATIO = 0.2
def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


# %%
#  TODO:  Implémenter

# Chargement du dataset et DataLoader
from torch.utils.data import TensorDataset

BATCH_SIZE = 300
N_ITERS = 1000
VAL_RATIO = 0.2   # Ratio du sous-ensemble d'entraînement utilisé pour la validation

def load_mnist_tensors():
    os.makedirs(DATA_PATH, exist_ok=True)
    mnist = fetch_openml("mnist_784", version=1, data_home=DATA_PATH, as_frame=False)
    X = mnist["data"].astype("float32") / 255.0   # Normalisation
    y = mnist["target"].astype("int64")

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # Partition : 60000 pour train, 10000 pour test
    train_X, train_y = X[:60000], y[:60000]
    test_X,  test_y  = X[60000:], y[60000:]

    # Sélection d'une petite partie : 5% du train, 20% du test
    n_train = int(len(train_X) * TRAIN_RATIO)
    n_test  = int(len(test_X) * TEST_RATIO)

    perm_train = torch.randperm(len(train_X))[:n_train]
    perm_test  = torch.randperm(len(test_X))[:n_test]

    train_X, train_y = train_X[perm_train], train_y[perm_train]
    test_X,  test_y  = test_X[perm_test],  test_y[perm_test]

    return train_X, train_y, test_X, test_y


def get_dataloaders():
    """Construit les DataLoader pour train, validation et test"""
    train_X, train_y, test_X, test_y = load_mnist_tensors()

    train_dataset_full = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    # On découpe encore 20% du train pour former un ensemble de validation
    n_val = int(len(train_dataset_full) * VAL_RATIO)
    n_train = len(train_dataset_full) - n_val
    train_dataset, val_dataset = random_split(train_dataset_full, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


from torch.utils.data import TensorDataset, Dataset
from torchvision import transforms


# Question 3: Transformations pour l'augmentation de données
class AddGaussianNoise(object):
    """Ajout d'un bruit gaussien sur l'image"""
    def __init__(self, mean=0.0, std=0.2):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # tensor: (1, 28, 28)
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class AugmentedMNISTDataset(Dataset):
    """Dataset MNIST avec possibilité d'appliquer des transformations"""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx]        # shape: (784,)
        y = self.labels[idx]

        # On remet l'image en (1,28,28) pour les transformations
        x = x.view(1, 28, 28)

        if self.transform is not None:
            x = self.transform(x)

        # On repasse en vecteur (784,) pour le MLP
        x = x.view(-1)

        return x, y


def get_dataloaders():
    """Version avec augmentation de données pour Q3"""
    train_X, train_y, test_X, test_y = load_mnist_tensors()

    # On sépare train en (train + validation) via un tirage aléatoire
    n_total = train_X.size(0)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # --- Transformations pour le jeu d'entraînement ---
    train_transform = transforms.Compose([
        # Les transforms torchvision s'attendent à (C,H,W)
        transforms.RandomRotation(15),            # petite rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),                # petites translations
            scale=(0.9, 1.1),                    # léger zoom
        ),
        AddGaussianNoise(mean=0.0, std=0.2),
    ])

    # Dataset avec augmentation pour TRAIN
    train_dataset = AugmentedMNISTDataset(
        train_X[train_idx],
        train_y[train_idx],
        transform=train_transform,
    )

    # Dataset SANS augmentation pour VALIDATION et TEST
    val_dataset = AugmentedMNISTDataset(
        train_X[val_idx],
        train_y[val_idx],
        transform=None,
    )

    test_dataset = AugmentedMNISTDataset(
        test_X,
        test_y,
        transform=None,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


# Modèle MLP avec options de régularisation
class MLP(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=100,
        num_classes=10,
        p_dropout=0.0,          # probabilité de dropout
        use_batchnorm=False,    # activer BatchNorm
        use_layernorm=False     # activer LayerNorm
    ):
        super().__init__()

        # Couches linéaires
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Classification finale
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Options de régularisation
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm

        # Dropout
        self.dropout = nn.Dropout(p_dropout)

        # BatchNorm / LayerNorm pour chaque couche cachée
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)

        if use_layernorm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Couche 1
        a1 = self.fc1(x)
        if self.use_batchnorm:
            a1 = self.bn1(a1)
        if self.use_layernorm:
            a1 = self.ln1(a1)
        a1 = F.relu(a1)
        a1 = self.dropout(a1)

        # Couche 2
        a2 = self.fc2(a1)
        if self.use_batchnorm:
            a2 = self.bn2(a2)
        if self.use_layernorm:
            a2 = self.ln2(a2)
        a2 = F.relu(a2)
        a2 = self.dropout(a2)

        # Couche 3
        a3 = self.fc3(a2)
        if self.use_batchnorm:
            a3 = self.bn3(a3)
        if self.use_layernorm:
            a3 = self.ln3(a3)
        a3 = F.relu(a3)
        a3 = self.dropout(a3)

        logits = self.classifier(a3)

        # Sauvegarde des activations pour TensorBoard
        activations = {
            "layer1": a1,
            "layer2": a2,
            "layer3": a3,
        }

        return logits, activations



# Fonctions utilitaires
def compute_entropy_from_logits(logits):
    """Calcule l'entropie de la distribution de sortie pour chaque échantillon"""
    probs = F.softmax(logits, dim=1)
    eps = 1e-12
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return entropy


def evaluate(model, loader, device, criterion):
    """Calcule la loss moyenne sur un DataLoader donné"""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
    return total_loss / max(1, n_samples)


# Boucle d'entraînement + enregistrement TensorBoard
def train_and_log():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # TensorBoard
    log_dir = "/Users/liukeyi/Desktop/Deep Learning/student_tp6/src/mnist-regularisation"  
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Enregistrer l'entropie d'un "modèle aléatoire" comme référence
    with torch.no_grad():
        rand_logits = torch.randn(1000, 10)
        rand_entropy = compute_entropy_from_logits(rand_logits)
        writer.add_histogram("entropy/random_model", rand_entropy, global_step=0)

    # On contrôle explicitement 1000 itérations
    train_iter = iter(train_loader)
    global_step = 0

    pbar = tqdm(total=N_ITERS, desc="Entraînement")

    while global_step < N_ITERS:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        # Aplatir en (batch, 784)
        x = x.view(x.size(0), -1)

        model.train()
        optimizer.zero_grad()

        logits, activations = model(x)

        # On applique store_grad pour récupérer les gradients sur les activations
        for name, act in activations.items():
            store_grad(act)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        global_step += 1
        pbar.update(1)

  
        # Loss scalaires (train/val/test)
        if global_step % 10 == 0 or global_step == 1:
            train_loss = loss.item()
            val_loss = evaluate(model, val_loader, device, criterion)
            test_loss = evaluate(model, test_loader, device, criterion)

            writer.add_scalar("loss/train", train_loss, global_step)
            writer.add_scalar("loss/val",   val_loss,   global_step)
            writer.add_scalar("loss/test",  test_loss,  global_step)

        # Histogrammes : poids, gradients, entropie
        if global_step % 50 == 0 or global_step == N_ITERS:
            # Histogrammes des poids
            for name, param in model.named_parameters():
                if "weight" in name:
                    writer.add_histogram(f"weights/{name}", param.detach().cpu(), global_step)

            # Gradients en entrée de chaque couche
            for name, act in activations.items():
                if act.grad is not None:
                    writer.add_histogram(f"grads/input_{name}", act.grad.detach().cpu(), global_step)

            # Entropie des sorties
            with torch.no_grad():
                entropy = compute_entropy_from_logits(logits)
                writer.add_histogram("entropy/output", entropy.cpu(), global_step)

    pbar.close()
    writer.close()
    logging.info("Entraînement terminé. Lancez : tensorboard --logdir runs")


# Point d'entrée principal
def main():
    train_and_log()


if __name__ == "__main__":
    main()

# %%

# %%
