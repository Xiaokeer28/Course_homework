# %%
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List

import time
import re
from torch.utils.tensorboard import SummaryWriter




logging.basicConfig(level=logging.INFO)

FILE = "/Users/liukeyi/Desktop/Deep Learning/student_tp5/data/en-fra.txt"

writer = SummaryWriter("/tmp/runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)


# %%
#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
EMB_DIM = 256
HID_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 8
GRAD_CLIP = 1.0
MAX_GEN_LEN = 60    
TF_PROB_PER_BATCH = 0.5  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.0, pad_idx=Vocabulary.PAD):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout if num_layers>1 else 0.0)

    def forward(self, src_TB, lengths_B):

        emb = self.embedding(src_TB)               # (T,B,E)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths_B.cpu(), enforce_sorted=False)
        _, hidden = self.gru(packed)               # hidden: (L,B,H)
        return hidden

# Decoder 
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.0, pad_idx=Vocabulary.PAD):
        super().__init__()
        self.sos = Vocabulary.SOS
        self.eos = Vocabulary.EOS
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout if num_layers>1 else 0.0)
        self.proj = nn.Linear(hid_dim, vocab_size)

    def forward_step(self, inp_B, hidden):
        emb = self.embedding(inp_B.unsqueeze(0))   
        out, hidden = self.gru(emb, hidden)        
        logits = self.proj(out.squeeze(0))        
        return logits, hidden

    @torch.no_grad()
    def generate(self, hidden, lenseq=None):
        B = hidden.size(1)
        assert B == 1
        max_len = int(lenseq) if lenseq is not None else MAX_GEN_LEN

        inp = torch.tensor([self.sos], device=hidden.device)  # (1,)
        out_tokens = []

        for _ in range(max_len):
            logits, hidden = self.forward_step(inp, hidden)
            next_tok = logits.argmax(-1)          # (1,)
            tok_id = next_tok.item()
            out_tokens.append(tok_id)
            if tok_id == self.eos:
                break
            inp = next_tok
        return out_tokens

# Seq2Seq wrapper 
class Seq2Seq(nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src_TB, src_len_B, trg_TB, teacher_forcing: bool):
        hidden = self.enc(src_TB, src_len_B)                   # (L,B,H)

        T_trg, B = trg_TB.shape
        logits_list = []
        inp = torch.full((B,), fill_value=Vocabulary.SOS, dtype=torch.long, device=trg_TB.device)

        for t in range(T_trg):
            logits_BC, hidden = self.dec.forward_step(inp, hidden)  
            logits_list.append(logits_BC.unsqueeze(0))               

            if teacher_forcing and t+1 < T_trg:
                inp = trg_TB[t]                                    
            else:
                inp = logits_BC.argmax(-1)

        logits_TBC = torch.cat(logits_list, dim=0)  # (T_trg,B,C)
        return logits_TBC

# Instantiate & training bits 
enc = Encoder(len(vocEng), EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT).to(device)
dec = Decoder(len(vocFra), EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT).to(device)
model = Seq2Seq(enc, dec).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
optimizer = optim.AdamW(model.parameters(), lr=LR)

def run_epoch(dataloader, train: bool):
    model.train(train)
    total_loss, total_tok = 0.0, 0
    for (src_TB, src_len_B, trg_TB, trg_len_B) in dataloader:
        src_TB, src_len_B = src_TB.to(device), src_len_B.to(device)
        trg_TB, trg_len_B = trg_TB.to(device), trg_len_B.to(device)

        use_tf = (torch.rand(1).item() < TF_PROB_PER_BATCH)

        logits_TBC = model(src_TB, src_len_B, trg_TB, teacher_forcing=use_tf)   # (T,B,C)
        T, B, C = logits_TBC.shape
        loss = criterion(logits_TBC.view(T*B, C), trg_TB.view(T*B))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        with torch.no_grad():
            mask = (trg_TB != Vocabulary.PAD)
            total_tok += mask.sum().item()
            total_loss += loss.item()

    return total_loss / max(1, len(dataloader))

best = 1e9
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    train_loss = run_epoch(train_loader, train=True)
    test_loss  = run_epoch(test_loader,  train=False)

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/test",  test_loss,  epoch)

    logging.info(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | {time.time()-t0:.1f}s")
    if test_loss < best:
        best = test_loss
        torch.save(model.state_dict(), "best_seq2seq_gru.pt")
        logging.info("Checkpoint saved.")

# check
@torch.no_grad()
def translate_sample(sample_text="i am happy ."):
    src_ids = torch.tensor([vocEng.get(w, adding=False) for w in normalize(sample_text).split()] + [Vocabulary.EOS], dtype=torch.long)
    src_TB = src_ids.unsqueeze(1).to(device)     
    src_len = torch.tensor([len(src_ids)], device=device)

    hidden = enc(src_TB, src_len)                 # (L,1,H)
    out_ids = dec.generate(hidden, lenseq=MAX_GEN_LEN)  # list[int]
    words = []
    for i in out_ids:
        if i == Vocabulary.EOS:
            break
        words.append(vocFra.getword(i))
    return " ".join([w for w in words if w is not None])

print("\n>>> demo:", translate_sample("i am happy ."))

# %%
# Question 3

import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import random

SPM_VOCAB = 8000
SPM_TYPE  = "bpe"       
PAD_ID    = 0
EMB_DIM   = 256
HID_DIM   = 512
BATCH_SIZE= 64
EPOCHS    = 1           
MAX_LEN   = 80
GEN_LEN   = 60
LR        = 1e-3
GRAD_CLIP = 1.0
TF_PROB   = 0.5
DATA_DIR  = "/Users/liukeyi/Desktop/Deep Learning/student_tp5/data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SentencePiece
def train_or_load_spm(all_lines):
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    en_model = data_dir / "spm_en.model"
    fr_model = data_dir / "spm_fr.model"
    if not (en_model.exists() and fr_model.exists()):
        print("Training SentencePiece models")
        en_txt = data_dir / "all_en.txt"
        fr_txt = data_dir / "all_fr.txt"
        with en_txt.open("w", encoding="utf-8") as fe, fr_txt.open("w", encoding="utf-8") as ff:
            for s in all_lines:
                s = s.strip()
                if not s: continue
                en, fr = s.split("\t")[:2]
                fe.write(en.strip() + "\n")
                ff.write(fr.strip() + "\n")
        spm.SentencePieceTrainer.train(
            input=str(en_txt), model_prefix=str(data_dir/"spm_en"),
            vocab_size=SPM_VOCAB, model_type=SPM_TYPE, character_coverage=1.0
        )
        spm.SentencePieceTrainer.train(
            input=str(fr_txt), model_prefix=str(data_dir/"spm_fr"),
            vocab_size=SPM_VOCAB, model_type=SPM_TYPE, character_coverage=1.0
        )
        print("SPM training complete.")
    sp_en = spm.SentencePieceProcessor(model_file=str(en_model))
    sp_fr = spm.SentencePieceProcessor(model_file=str(fr_model))
    return sp_en, sp_fr


# Dataset
class SPMTradDataset(torch.utils.data.Dataset):
    def __init__(self, lines, sp_src, sp_tgt, max_len=100):
        self.samples = []
        for s in lines:
            s = s.strip()
            if not s: continue
            src, tgt = s.split("\t")[:2]
            src_ids = sp_src.encode(src, out_type=int) + [sp_src.eos_id()]
            tgt_ids = sp_tgt.encode(tgt, out_type=int) + [sp_tgt.eos_id()]
            if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                self.samples.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_fn_sp(batch):
    src, tgt = zip(*batch)
    src_len = torch.tensor([len(s) for s in src])
    tgt_len = torch.tensor([len(t) for t in tgt])
    src_pad = pad_sequence(src, padding_value=PAD_ID)
    tgt_pad = pad_sequence(tgt, padding_value=PAD_ID)
    return src_pad, src_len, tgt_pad, tgt_len


# Decoder
class DecoderSPM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, sos_id, eos_id,
                 num_layers=1, dropout=0.0, pad_idx=PAD_ID):
        super().__init__()
        self.sos = sos_id
        self.eos = eos_id
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers=num_layers,
                          dropout=dropout if num_layers>1 else 0.0)
        self.proj = nn.Linear(hid_dim, vocab_size)

    def forward_step(self, inp_B, hidden):
        emb = self.embedding(inp_B.unsqueeze(0))
        out, hidden = self.gru(emb, hidden)
        logits = self.proj(out.squeeze(0))
        return logits, hidden

    @torch.no_grad()
    def generate(self, hidden, lenseq=None):
        B = hidden.size(1)
        assert B == 1
        max_len = int(lenseq) if lenseq is not None else GEN_LEN
        inp = torch.tensor([self.sos], device=hidden.device)
        out_tokens = []
        for _ in range(max_len):
            logits, hidden = self.forward_step(inp, hidden)
            nxt = logits.argmax(-1)
            tid = nxt.item()
            out_tokens.append(tid)
            if tid == self.eos:
                break
            inp = nxt
        return out_tokens


with open(FILE, encoding="utf-8") as f:
    all_lines = f.readlines()
random.shuffle(all_lines)
cut = int(0.8*len(all_lines))
train_lines = all_lines[:cut]
test_lines  = all_lines[cut:]

sp_en, sp_fr = train_or_load_spm(all_lines)
ds_train = SPMTradDataset(train_lines, sp_en, sp_fr, max_len=MAX_LEN)
ds_test  = SPMTradDataset(test_lines,  sp_en, sp_fr, max_len=MAX_LEN)

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn_sp)
dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_sp)

Vsrc = sp_en.vocab_size()
Vtgt = sp_fr.vocab_size()
sos_id, eos_id = sp_en.bos_id(), sp_en.eos_id()

print(f"[Q3] train {len(ds_train)}, test {len(ds_test)}, Vsrc {Vsrc}, Vtgt {Vtgt}")


# modele
enc_q3 = Encoder(Vsrc, EMB_DIM, HID_DIM).to(device)
dec_q3 = DecoderSPM(Vtgt, EMB_DIM, HID_DIM, sos_id, eos_id).to(device)
model_q3 = Seq2Seq(enc_q3, dec_q3).to(device)

criterion_q3 = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer_q3 = optim.AdamW(model_q3.parameters(), lr=LR)


# entrainement
def run_epoch_q3(loader, train=True):
    model_q3.train(train)
    total = 0.0
    for step, (src_TB, src_len_B, tgt_TB, tgt_len_B) in enumerate(loader):
        src_TB, src_len_B = src_TB.to(device), src_len_B.to(device)
        tgt_TB = tgt_TB.to(device)
        use_tf = (random.random() < TF_PROB)
        logits = model_q3(src_TB, src_len_B, tgt_TB, teacher_forcing=use_tf)
        T, B, V = logits.shape
        loss = criterion_q3(logits.view(T*B, V), tgt_TB.view(T*B))
        if train:
            optimizer_q3.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model_q3.parameters(), GRAD_CLIP)
            optimizer_q3.step()
        total += loss.item()
    return total / max(1, (step+1))


best = 1e9
for ep in range(1, EPOCHS+1):
    t0 = time.time()
    tr = run_epoch_q3(dl_train, True)
    te = run_epoch_q3(dl_test, False)
    print(f"[Q3] Epoch {ep:02d} | train_loss={tr:.4f} | test_loss={te:.4f} | {time.time()-t0:.1f}s")
    if te < best:
        best = te
        torch.save(model_q3.state_dict(), Path(DATA_DIR)/"best_seq2seq_q3.pt")
        print("[Q3] checkpoint saved")


# visualsation
@torch.no_grad()
def translate_q3(text="I am happy ."):
    src_ids = sp_en.encode(text, out_type=int) + [sp_en.eos_id()]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_ids)], device=device)
    h = enc_q3(src, src_len)
    out_ids = dec_q3.generate(h, lenseq=GEN_LEN)
    return sp_fr.decode(out_ids)

print("\n[Q3] demo:", translate_q3("I am happy ."))

# %%
