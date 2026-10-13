# %%
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *


DATA_PATH = "/Users/liukeyi/Desktop/Deep Learning/student_tp3/data/trump_full_speech.txt"


# %%
# Question 2
def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    length, batch, output_dim = output.shape
    output = output.reshape(-1, output_dim)    
    target = target.reshape(-1)                
    mask = (target != padcar).float()  # Create mask where target != padcar (1 for valid, 0 for padding)
    loss_per_char = nn.functional.cross_entropy(output, target, reduction='none')   
    loss_masked = loss_per_char * mask # ignore padded positions
    mean_loss = loss_masked.sum() / mask.sum()

    return mean_loss


# %%
class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)  # Embedding
        self.rnn = nn.GRU(emb_dim, hidden_size)            
        self.decoder = nn.Linear(hidden_size, vocab_size)   

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        emb = self.embedding(input)        # (seq_len, batch_size, emb_dim)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)     # (seq_len, batch_size, vocab_size)
        return decoded, hidden


# %% 
# Question 4
class LSTM(nn.Module):
    #  TODO:  Implémenter un LSTM
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden=None):
        """
        Pour LSTM, hidden est un tuple (h, c)
        """
        emb = self.embedding(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)
        return decoded, hidden

# %%
class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden=None):
        """
        input : (seq_len, batch)
        hidden : (1, batch, hidden_size)
        """
        emb = self.embedding(input)                 # (seq_len, batch, emb_dim)
        output, hidden = self.rnn(emb, hidden)      # output: (seq_len, batch, hidden_size)
        decoded = self.decoder(output)              # (seq_len, batch, vocab_size)
        return decoded, hidden



# %%
vocab_size = len(lettre2id)
emb_dim = 64
hidden_size = 128
epochs = 10
learning_rate = 0.01

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()
dataset = TextDataset(text, maxsent=5000)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)

models_to_compare = {
    "GRU": GRU(vocab_size, emb_dim, hidden_size),
    "LSTM": LSTM(vocab_size, emb_dim, hidden_size)
}
run_name = "Comparison"

for model_name, model in models_to_compare.items():
    
    print(f"\n--- Entraînement du modèle : {model_name} ---")
    
    writer = SummaryWriter(log_dir=f'runs/{run_name}/{model_name}') # Initialiser un writer TensorBoard
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    global_step = 0 # Compteur pour les logs par batch
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in loader:
            if batch is None: continue 
                
            x_batch = batch[:-1, :]
            y_batch = batch[1:, :]

            optimizer.zero_grad()
            output, hidden = model(x_batch)          
            loss = maskedCrossEntropy(output, y_batch, padcar=PAD_IX)         
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            
            writer.add_scalar(f'Loss/train_batch', loss.item(), global_step)
            global_step += 1

        # Calcul et affichage de la loss moyenne de l'époque
        epoch_loss = total_loss / len(loader)
        print(f"Modèle {model_name} - Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
        writer.add_scalar(f'Loss/train_epoch', epoch_loss, epoch)
    
    writer.close()
    print(f"--- Fin de l'entraînement pour {model_name} ---")

print("\nComparaison terminée.")
print(f"Pour voir les résultats, lancez : tensorboard --logdir=runs/{run_name}")


# %%
# Comparaison
print("Comparaison des générations:")

# Charger les modèles entraînés
trained_gru = models_to_compare["GRU"]
trained_lstm = models_to_compare["LSTM"]

prompts = ["Trump", "America", "We"] # Choisir les prompts à tester

maxlen = 120

# Boucle de comparaison
for prompt in prompts:
    print(f"Prompt : '{prompt}'")

    text_greedy_gru = generate_beam(trained_gru, EOS_IX, k=1, start=prompt, maxlen=maxlen)
    print("GRU (k=1, Greedy) :")
    print(text_greedy_gru)

    text_beam_gru = generate_beam(trained_gru, EOS_IX, k=5, start=prompt, maxlen=maxlen)
    print("GRU (k=5, Beam Search) :")
    print(text_beam_gru)

    text_beam_lstm = generate_beam(trained_lstm, EOS_IX, k=5, start=prompt, maxlen=maxlen)
    print("LSTM (k=5, Beam Search) :")
    print(text_beam_lstm)

print("Fin de la comparaison")


# %%
# Question 7 
@torch.no_grad()
def generate_beam_nucleus(model, eos, k=5, alpha=0.9, start="", maxlen=200):
    """
    Génère une séquence avec beam search utilisant nucleus sampling (Question 7)
    
    Args:
        model : modèle complet (RNN, GRU ou LSTM)
        eos : index du token EOS
        k : beam width
        alpha : masse de probabilité nucleus (ex. 0.9)
        start : début de phrase
        maxlen : longueur maximale
    """
    model.eval()
    h = None

    # Initialisation du début de séquence
    if start == "":
        seq = torch.tensor([PAD_IX], dtype=torch.long)
    else:
        seq = string2code(start)

    # Si prompt long, initialiser le hidden state
    if len(seq) > 1:
        prompt_seq = seq[:-1].unsqueeze(1)
        emb_prompt = model.embedding(prompt_seq)
        _, h = model.rnn(emb_prompt, None)

    nucleus = p_nucleus(model.decoder, alpha)
    beams = [(seq, 0.0, h)]

    for _ in range(maxlen):
        candidates = []
        all_finished = True

        for seq, log_p, h in beams:
            if seq[-1].item() == eos:
                candidates.append((seq, log_p, h))
                continue

            all_finished = False
            last_token = seq[-1].unsqueeze(0).unsqueeze(0)
            emb_in = model.embedding(last_token)
            out, h_new = model.rnn(emb_in, h)

            probs = nucleus(out[-1])  # (vocab_size,)
            token = torch.multinomial(probs, 1).item()
            new_log_p = log_p + torch.log(probs[token] + 1e-8).item()
            new_seq = torch.cat([seq, torch.tensor([token])])
            candidates.append((new_seq, new_log_p, h_new))

        if all_finished:
            break

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

    best_seq, _, _ = beams[0]
    tokens = [t for t in best_seq.tolist() if t not in (PAD_IX, EOS_IX)]
    return code2string(tokens)


# %%
print("\nBeam Search + Nucleus Sampling")

gru_model = models_to_compare["GRU"]
lstm_model = models_to_compare["LSTM"]

print("\nGRU (start='Trump'):")
print(generate_beam_nucleus(gru_model, EOS_IX, k=3, alpha=0.9, start="Trump", maxlen=120))

print("\nLSTM (start='America'):")
print(generate_beam_nucleus(lstm_model, EOS_IX, k=3, alpha=0.9, start="America", maxlen=120))



# %%
