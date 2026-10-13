# %%:
# Question 1
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, dim, latent, output):
        super().__init__()
        self.Wi = nn.Linear(dim, latent)
        self.Wh = nn.Linear(latent, latent)
        self.Wd = nn.Linear(latent, output)

    def one_step(self, x, h):
        return torch.tanh(self.Wi(x) + self.Wh(h))  #torch.tanh(x @ self.Wi + h @ self.Wh + self.bh)

    def forward(self, x):
        # x: [length, batch, dim]
        length, batch, dim = x.size()
        h = torch.zeros(batch, self.Wh.out_features, device=x.device)
        hs = []
        for t in range(length):
            h = self.one_step(x[t], h)
            hs.append(h)
        return torch.stack(hs, dim=0)  # [length, batch, latent]

    def decode(self, h):
        return self.Wd(h)  # [batch, n_classes]


# %%:
# Question 2
from utils import SampleMetroDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Paramètres
CLASSES = 10           # nombre de stations
LENGTH = 20            # longueur des séquences
DIM_INPUT = 2          # 1 (entrée) ou 2 (entrée/sortie)
BATCH_SIZE = 32
PATH = r"/Users/liukeyi/Desktop/Deep Learning/student_tp3/data/"

matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)

data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

LATENT = 64
EPOCHS = 15
LR = 1e-3

model = RNN(dim=DIM_INPUT, latent=LATENT, output=CLASSES)
criterion = nn.CrossEntropyLoss()                 
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
grad_clip = 5.0                               

# Boucle d'entraînement et de test
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, total_correct, total_n = 0, 0, 0

    for X, y in loader:
        # X: [length, batch, dim], y: [batch]
        X = X.permute(1, 0, 2)
        y = y.long()

        hs = model.forward(X)          # [length, batch, latent]
        h_T = hs[-1]                   # dernier état caché
        logits = model.decode(h_T)     # [batch, classes]
        loss = criterion(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # calcul des statistiques
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_n += y.numel()
            total_loss += loss.item() * y.numel()

    return total_loss / max(1, total_n), total_correct / max(1, total_n)

# Entraînement
best_acc = 0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(data_train, train=True)
    te_loss, te_acc = run_epoch(data_test, train=False)
    best_acc = max(best_acc, te_acc)
    print(f"Epoch {epoch:02d} | "
          f"Train Loss {tr_loss:.3f} Acc {tr_acc:.3f} | "
          f"Test Loss {te_loss:.3f} Acc {te_acc:.3f}")

print(f"Meilleure précision en test : {best_acc:.3f}")


# %%
# Question 3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import ForecastMetroDataset
import matplotlib.pyplot as plt

# Paramètres
CLASSES = 10
LENGTH = 20
DIM_INPUT = 2
BATCH_SIZE = 32
LATENT = 64
EPOCHS = 20
LR = 1e-3
PATH = "/Users/liukeyi/Desktop/Deep Learning/student_tp3/data/"

# Données
matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

# Modèle
class RNNForecast(nn.Module):
    def __init__(self, dim, latent, output):
        super().__init__()
        self.Wi = nn.Linear(dim, latent)
        self.Wh = nn.Linear(latent, latent)
        self.Wd = nn.Linear(latent, output)
    def one_step(self, x, h):
        return torch.tanh(self.Wi(x) + self.Wh(h))
    def forward(self, x):
        length, batch, _ = x.size()
        h = torch.zeros(batch, self.Wh.out_features)
        ys = []
        for t in range(length):
            h = self.one_step(x[t], h)
            y = self.Wd(h)
            ys.append(y)
        return torch.stack(ys, dim=0)

# Entraînement
model = RNNForecast(dim=CLASSES * DIM_INPUT, latent=LATENT, output=CLASSES * DIM_INPUT)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def run_epoch(loader, train=True):
    model.train(train)
    total_loss, total_samples = 0.0, 0
    for xb, yb in loader:
        B, L, S, D = xb.shape
        xb = xb.reshape(B, L, S * D).permute(1, 0, 2)
        yb = yb.reshape(B, L, S * D).permute(1, 0, 2)
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(1)
        total_samples += xb.size(1)
    return total_loss / max(1, total_samples)

# Boucle principale
train_losses, test_losses = [], []
best_loss = float('inf')
for epoch in range(1, EPOCHS + 1):
    tr_loss = run_epoch(data_train, train=True)
    te_loss = run_epoch(data_test, train=False)
    best_loss = min(best_loss, te_loss)
    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    print(f"Epoch {epoch:02d} | Train Loss {tr_loss:.6f} | Test Loss {te_loss:.6f}")

print(f"Meilleure perte en test : {best_loss:.6f}")

# Visualisation
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(test_losses, label='Test Loss', marker='s')
plt.title("Perte (MSE) selon les époques")
plt.xlabel("Époque")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Question 4 — Trump text generation (version complète CPU)

import string, unicodedata, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PATH = "/Users/liukeyi/Desktop/Deep Learning/student_tp3/data/"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
EMBED_DIM = 64
LATENT = 256

device = torch.device("cpu")

# Jeu de caractères et encodage
LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[0] = ''
lettre2id = {v: k for k, v in id2lettre.items()}
VOCAB_SIZE = len(lettre2id)

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)], dtype=torch.long)

def code2string(t):
    if not isinstance(t, list):
        t = t.tolist()
    return ''.join(id2lettre.get(i, '') for i in t)

# Dataset de texte Trump
class TrumpDataset(Dataset):
    def __init__(self, text, maxsent=None, maxlen=None):
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip() + '.' for p in full_text.split(".") if len(p) > 0]
        if maxsent is not None:
            self.phrases = self.phrases[:maxsent]
        self.MAX_LEN = max(len(p) for p in self.phrases)

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN - t.size(0), dtype=torch.long), t])
        return t[:-1], t[1:]  # entrée, cible

# RNN
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.Wi = nn.Linear(embed_dim, latent_dim)
        self.Wh = nn.Linear(latent_dim, latent_dim)
        self.Wd = nn.Linear(latent_dim, vocab_size)

    def one_step(self, x, h):
        x = self.embed(x)
        return torch.tanh(self.Wi(x) + self.Wh(h))

    def forward(self, x):
        seq_len, batch = x.size()
        h = torch.zeros(batch, self.Wh.out_features)
        ys = []
        for t in range(seq_len):
            h = self.one_step(x[t], h)
            y = self.Wd(h)
            ys.append(y)
        return torch.stack(ys, dim=0)


# Chargement des données
with open(PATH + "trump_full_speech.txt", "rb") as f:
    text = f.read().decode()

dataset = TrumpDataset(text, maxsent=2000, maxlen=200)
data_trump = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialisation du modèle et optimiseur
model = RNN(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, latent_dim=LATENT)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Entraînement
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in data_trump:
        xb, yb = xb.t(), yb.t()  # [seq_len, batch]
        logits = model(xb)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d} | Loss: {total_loss / len(data_trump):.4f}")

print("Entraînement terminé.")

# Génération de texte
def top_k_logits(logits, k=40):
    v, _ = torch.topk(logits, k)
    thresh = v[:, -1].unsqueeze(1)
    return torch.where(logits < thresh, torch.full_like(logits, -1e10), logits)

def generate(model, start_text="The ", length=400, temperature=0.8, topk=40):
    model.eval()
    h = torch.zeros(1, model.Wh.out_features)
    for ch in start_text[:-1]:
        x = torch.tensor([[lettre2id.get(ch, 0)]])
        h = model.one_step(x[0], h)
    output = start_text
    x = torch.tensor([[lettre2id.get(start_text[-1], 0)]])
    for _ in range(length):
        h = model.one_step(x[0], h)
        logits = model.Wd(h) / temperature
        if topk is not None:
            logits = top_k_logits(logits, k=topk)
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        output += id2lettre.get(idx, '')
        x = torch.tensor([[idx]])
    return output


print("\nTexte généré :\n")
print(generate(model, start_text="The ", length=400))

# %%
