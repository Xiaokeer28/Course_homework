# %%
import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

from tp7_preprocess import TextDataset
import torch.serialization
torch.serialization.add_safe_globals([TextDataset])

# Utiliser tp7_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path("/Users/liukeyi/Desktop/Deep Learning/student_tp7/")

# Chargement du tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(str(MAINDIR / f"wp{vocab_size}.model"))
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(MAINDIR / f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp, weights_only=False)



train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
test_size = 10000
train_size = len(train) - val_size -test_size
train, val, test = torch.utils.data.random_split(train, [train_size, val_size,test_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


#  TODO: 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

num_classes = 2 


def compute_majority_class(dataloader, num_classes: int = 2):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in dataloader:
        labels = batch.labels
        counts += torch.bincount(labels, minlength=num_classes)
    majority = counts.argmax().item()
    logging.info(f"Label counts on train: {counts.tolist()}, majority={majority}")
    return majority

def eval_majority_baseline(majority_class: int, dataloader):
    correct = 0
    total = 0
    for batch in dataloader:
        labels = batch.labels
        preds = torch.full_like(labels, fill_value=majority_class)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total

majority_class = compute_majority_class(train_iter, num_classes=num_classes)
baseline_test_acc = eval_majority_baseline(majority_class, test_iter)
logging.info(f"Majority baseline test accuracy = {baseline_test_acc:.4f}")



class CNNText(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int = 128,
                 num_filters: int = 100,
                 kernel_size: int = 3,
                 num_classes: int = 2,
                 padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(
            in_channels=emb_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)       # (batch, seq_len, emb_dim)
        emb = emb.transpose(1, 2)     # (batch, emb_dim, seq_len)
        h = torch.relu(self.conv(emb))  # (batch, num_filters, seq_len)
        pool = MaxPool1d(kernel_size=h.size(2))
        h = pool(h).squeeze(-1)       # (batch, num_filters)
        out = self.fc(h)              # (batch, num_classes)
        return out


def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    for step, batch in enumerate(dataloader):
        inputs = batch.text.to(device)
        targets = batch.labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (preds == targets).sum().item()
        total_examples += targets.size(0)

        if step % 20 == 0:
            acc = (preds == targets).float().mean().item()
            logging.info(f"[train] epoch={epoch} step={step} loss={loss.item():.4f} acc={acc:.4f}")

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def evaluate(model, dataloader, criterion=None, mode="val", epoch=0):
    model.eval()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    for batch in dataloader:
        inputs = batch.text.to(device)
        targets = batch.labels.to(device)
        logits = model(inputs)
        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_examples += targets.size(0)

    acc = total_correct / total_examples
    if criterion is not None:
        loss = total_loss / total_examples
        logging.info(f"[{mode}] epoch={epoch} loss={loss:.4f} acc={acc:.4f}")
    else:
        logging.info(f"[{mode}] acc={acc:.4f}")
    return acc

EPOCHS = 3 

model = CNNText(
    vocab_size=ntokens,
    emb_dim=128,
    num_filters=100,
    kernel_size=3,
    num_classes=num_classes,
    padding_idx=0
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_acc = 0.0
best_state = None

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_iter, optimizer, criterion, epoch)
    logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
    val_acc = evaluate(model, val_iter, criterion, mode="val", epoch=epoch)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

if best_state is not None:
    model.load_state_dict(best_state)
test_acc = evaluate(model, test_iter, mode="test")

logging.info(f"Majority baseline test acc = {baseline_test_acc:.4f}")
logging.info(f"CNN best val acc = {best_val_acc:.4f}, test acc = {test_acc:.4f}")

# %%
