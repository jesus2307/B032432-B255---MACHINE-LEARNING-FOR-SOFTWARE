import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import CharRNN

data_file = "data/processed/processed_data.txt"

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file {filepath} does not exist.")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.readlines()
    text = ''.join(text[:10000])  # Usar hasta 10,000 líneas
    chars = sorted(set(text))
    char2int = {ch: i for i, ch in enumerate(chars)}
    int2char = {i: ch for ch, i in char2int.items()}
    return chars, char2int, int2char, text

def train(net, data, char2int, epochs=10, n_seqs=32, n_steps=32, lr=0.001, clip=5, val_frac=0.1, device=torch.device('cpu'),
          name='models/training-datos.pth', early_stop=True, plot=True):
    print("Starting training...")
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    net.to(device)
    min_val_loss = float('inf')
    
    for e in range(epochs):
        hidden = None
        for x, y in get_batches(data, n_seqs, n_steps, char2int):
            x = one_hot_encode(x, len(net.chars))
            inputs, targets = torch.from_numpy(x).to(device), torch.tensor(y, dtype=torch.long).to(device)
            targets = torch.clamp(targets, min=0, max=len(net.chars) - 1)

            net.zero_grad()
            output, hidden = net.forward(inputs, hidden)
            loss = criterion(output.view(n_seqs * n_steps, len(net.chars)), targets.view(n_seqs * n_steps))
            loss.backward()
            
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            if hidden is not None:
                hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()

        print(f"Epoch {e+1}/{epochs} - Loss: {loss.item():.4f}")

        # Guardar el mejor modelo
        if loss.item() < min_val_loss:
            min_val_loss = loss.item()
            save_checkpoint(net, opt, name)
            print(f"✔ Modelo guardado en {name} con pérdida {min_val_loss:.4f}")

def get_batches(data, n_seqs, n_steps, char2int):
    batch_size = n_seqs * n_steps
    n_batches = len(data) // batch_size
    data = data[:n_batches * batch_size]
    data = np.array([char2int[c] for c in data]).reshape((n_seqs, -1))
    
    for n in range(0, data.shape[1] - 1, n_steps):
        x = data[:, n:n + n_steps]
        y = data[:, n + 1:n + n_steps + 1]
        if y.shape[1] != x.shape[1]:
            continue
        yield x, y

def one_hot_encode(arr, n_labels):
    arr = np.clip(arr, 0, n_labels - 1)
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    return one_hot.reshape((*arr.shape, n_labels))

def save_checkpoint(net, opt, filename, train_history={}):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict(),
                  'tokens': net.chars,
                  'train_history': train_history}
    torch.save(checkpoint, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename, map_location='cpu')
    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])
    return net, checkpoint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Cargar datos procesados
chars, char2int, int2char, data = load_data(data_file)

# Inicializar modelo
net = CharRNN(chars, n_hidden=128, n_layers=3)

# Entrenamiento
plt.figure(figsize=(12, 4))
train(net, data, char2int, epochs=10, n_seqs=32, n_steps=32, lr=0.001, device=device, val_frac=0.1,
      name='models/training-datos.pth', plot=True)
