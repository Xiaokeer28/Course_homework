# %%
import itertools
import logging
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse_incr
logging.basicConfig(level=logging.INFO)

DATA_PATH = "/Users/liukeyi/Desktop/Deep Learning/student_tp5/data/"


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
                self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))

logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)

data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu",encoding="utf-8")
train_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu",encoding='utf-8')
dev_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu",encoding="utf-8")
test_data = TaggingDataset(parse_incr(data_file), words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)




# %%
#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)

EMB_DIM = 200
HID_DIM = 256
NUM_LAYERS = 1
BIDIR = True
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 8
GRAD_CLIP = 1.0
OOV_PROB = 0.15     
USE_PACKING = True  
LOG_INTERVAL = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Utilities 
def lengths_from_padded(padded: torch.Tensor, pad_idx: int = Vocabulary.PAD):
    return (padded != pad_idx).sum(dim=0)

def replace_with_oov(batch_tokens: torch.Tensor, prob: float, oov_id: int, pad_idx: int):
    if prob <= 0:
        return batch_tokens
    with torch.no_grad():
        mask_valid = (batch_tokens != pad_idx)
        rand = torch.rand_like(batch_tokens, dtype=torch.float)
        flip = (rand < prob) & mask_valid
        out = batch_tokens.clone()
        out[flip] = oov_id
        return out

# Model
class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tag_size, emb_dim, hid_dim, num_layers=1,
                 bidir=True, dropout=0.0, pad_idx=Vocabulary.PAD):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir,
            batch_first=False
        )
        out_dim = hid_dim * (2 if bidir else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, tag_size)

    def forward(self, src_TB, lengths_B, use_packing=True):
        emb = self.embedding(src_TB)  # (T,B,emb)
        if use_packing:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths_B.cpu(), enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out_TBN, _ = nn.utils.rnn.pad_packed_sequence(packed_out)  # (T,B,H*)
        else:
            out_TBN, _ = self.lstm(emb)

        out_TBN = self.dropout(out_TBN)
        logits_TBN = self.classifier(out_TBN)  # (T,B,|tags|)
        return logits_TBN

# Instantiate 
model = LSTMTagger(
    vocab_size=len(words),
    tag_size=len(tags),
    emb_dim=EMB_DIM,
    hid_dim=HID_DIM,
    num_layers=NUM_LAYERS,
    bidir=BIDIR,
    dropout=DROPOUT,
    pad_idx=Vocabulary.PAD
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD) 
optimizer = optim.AdamW(model.parameters(), lr=LR)
writer = SummaryWriter() 

# Train / Eval loops
def run_epoch(dataloader, train: bool):
    model.train(train)
    total_loss, total_tok, correct_tok = 0.0, 0, 0

    for step, (x_TB, y_TB) in enumerate(dataloader):
        x_TB = x_TB.to(device)  
        y_TB = y_TB.to(device) 
        lengths_B = lengths_from_padded(x_TB, Vocabulary.PAD)

        if train:
            x_in = replace_with_oov(x_TB, OOV_PROB, Vocabulary.OOVID, Vocabulary.PAD)
        else:
            x_in = x_TB

        logits_TBN = model(x_in, lengths_B, use_packing=USE_PACKING)  # (T,B,|Y|)
        T, B, C = logits_TBN.shape

        loss = criterion(logits_TBN.view(T * B, C), y_TB.view(T * B))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        with torch.no_grad():
            pred_TB = logits_TBN.argmax(-1)  # (T,B)
            mask = (y_TB != Vocabulary.PAD)
            correct_tok += (pred_TB[mask] == y_TB[mask]).sum().item()
            total_tok += mask.sum().item()

        total_loss += loss.item()

        if train and (step + 1) % LOG_INTERVAL == 0:
            logging.info(f"[train] step {step+1}/{len(dataloader)}  loss={total_loss/(step+1):.4f}")

    avg_loss = total_loss / max(1, len(dataloader))
    acc = correct_tok / max(1, total_tok)
    return avg_loss, acc

best_dev = 0.0
global_step = 0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, train=True)
    dev_loss, dev_acc = run_epoch(dev_loader, train=False)

    if writer is not None:
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/dev", dev_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/dev", dev_acc, epoch)

    logging.info(f"Epoch {epoch:02d} | "
                 f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                 f"dev_loss={dev_loss:.4f} acc={dev_acc:.4f} | "
                 f"time={time.time()-t0:.1f}s")

    if dev_acc > best_dev:
        best_dev = dev_acc
        torch.save(model.state_dict(), "best_lstm_tagger.pt")
        logging.info(f"Saved checkpoint with dev_acc={best_dev:.4f}")

# Test
model.load_state_dict(torch.load("best_lstm_tagger.pt", map_location=device))
test_loss, test_acc = run_epoch(test_loader, train=False)
logging.info(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

# Visualization
def predict_sentence(tokens_T1B):
    """tokens_T1B: (T,1)"""
    model.eval()
    with torch.no_grad():
        tokens_T1B = tokens_T1B.to(device)
        lengths_B = (tokens_T1B != Vocabulary.PAD).sum(dim=0)
        logits = model(tokens_T1B, lengths_B, use_packing=USE_PACKING)  # (T,1,|Y|)
        pred = logits.argmax(-1).cpu()  
    return pred

x_TB, y_TB = next(iter(test_loader))
b = 0
T = int((x_TB[:, b] != Vocabulary.PAD).sum())
x_one = x_TB[:T, b:b+1]                
pred = predict_sentence(x_one)      

tokens = words.getwords(x_one.squeeze(1).tolist())
pred_tags = tags.getwords(pred[:T, 0].tolist())

print("Sentence:", " ".join(tokens))
print("PredTags:", " ".join(pred_tags))
# %%


