# %%
from textloader import code2string, string2code, PAD_IX, EOS_IX, id2lettre
import math
import torch
import torch.nn as nn



#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    rnn.eval()
    generated = []

    hidden = None

    if start == "":
        current = torch.zeros(1, 1, dtype=torch.long, device=device)
    else:
        current = string2code(start)
        current = current[-1].view(1, 1).to(device) 
        generated.append(start)

    for _ in range(maxlen):
        emb_out = emb(current)
        output, hidden = rnn(emb_out, hidden)
        # Decoder
        logits = decoder(output.squeeze(0))  # shape (1, vocab_size)
        probs = nn.functional.softmax(logits, dim=-1)

        if sampling:
            next_id = torch.multinomial(probs, 1).item()
        else:
            next_id = torch.argmax(probs).item()

        if next_id == eos:
            break

        generated.append(id2lettre[next_id])
        current = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return "".join(generated)

# %%
# Qyestion 6
@torch.no_grad() 
def generate_beam(model, eos, k, start="", maxlen=200, min_len=10): # Ajout de min_len
    
    model.eval()
    h = None
    if start == "":
        seq = torch.tensor([PAD_IX], dtype=torch.long)
    else:
        seq = string2code(start)
    
    if len(seq) > 1:
        prompt_seq = seq[:-1].unsqueeze(1)
        emb_prompt = model.embedding(prompt_seq) 
        _, h = model.rnn(emb_prompt, None) 
        
    beams = [(seq, 0.0, h)]

    for i in range(maxlen): 
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
            logits = model.decoder(out[-1])
            probs = nn.functional.log_softmax(logits, dim=-1)

            current_len = i + len(seq) 
            if current_len < min_len:
                probs[0, eos] = -float('Inf')

                
            topk_log_p, topk_idx = torch.topk(probs, k)

            for j in range(k):
                token = topk_idx[0, j].item()
                new_log_p = log_p + topk_log_p[0, j].item()
                
                new_seq = torch.cat([seq, torch.tensor([token])])
                candidates.append((new_seq, new_log_p, h_new))

        if all_finished:
            break
            
        # Tri des candidats (sans normalisation pour l'instant)
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

    best_seq, _, _ = beams[0]
    return code2string(best_seq.tolist())

# %%
# Question 7 
# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
        logits = decoder(h)                         # (1, vocab_size)
        probs = nn.functional.softmax(logits, dim=-1).squeeze(0)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=0)   
        cutoff = (cum_probs <= alpha).sum().item() + 1 # Trouver combien de tokens couvrent la masse α

        probs_masked = torch.zeros_like(probs)
        probs_masked[sorted_idx[:cutoff]] = probs[sorted_idx[:cutoff]]

      
        probs_nucleus = probs_masked / probs_masked.sum()   # Normalisation

        return probs_nucleus
    return compute


# %%
