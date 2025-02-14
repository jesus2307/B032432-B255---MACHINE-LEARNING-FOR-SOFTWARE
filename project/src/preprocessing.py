import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import unidecode

# Definir directorios de datos
data_dir = "data/processed/"
data_file = "data/datos.txt"

# Crear el directorio de datos si no existe
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def clean_content(x):
    """
    Función de preprocesamiento para limpiar el contenido del archivo.
    Elimina comentarios, múltiples saltos de línea y caracteres no ASCII.
    
    Args:
        x (str): Texto de entrada.
    
    Returns:
        str: Texto preprocesado.
    """
    x = unidecode.unidecode(x)  # Elimina caracteres especiales
    x = re.sub('#.*$', '', x, flags=re.MULTILINE)  # Elimina comentarios en línea
    x = re.sub(r"'''[\s\S]*?'''", '', x, flags=re.MULTILINE)  # Elimina comentarios multilínea
    x = re.sub(r'"""[\s\S]*?"""', '', x, flags=re.MULTILINE)  # Elimina comentarios multilínea
    x = re.sub('^[\t]+\n', '', x, flags=re.MULTILINE)  # Elimina líneas con solo tabulaciones
    x = re.sub('^[ ]+\n', '', x, flags=re.MULTILINE)  # Elimina líneas con solo espacios
    x = re.sub('\n[\n]+', '\n\n', x, flags=re.MULTILINE)  # Elimina saltos de línea extra
    x += '\nEOF\n'  # Añade marcador de final de archivo
    return x

# Leer y procesar datos desde el archivo si existe
if os.path.exists(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = f.read()
    cleaned_data = clean_content(raw_data)
    
    # Guardar datos preprocesados
    processed_file = os.path.join(data_dir, "processed_data.txt")
    with open(processed_file, "w", encoding="utf-8") as f:
        f.write(cleaned_data)
    print(f"Datos preprocesados guardados en {processed_file}")
else:
    print(f"El archivo {data_file} no existe.")

class CharRNN(nn.Module):
    """
    Implementación de una red neuronal recurrente basada en LSTM para generar texto carácter a carácter.
    """
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        """
        Paso hacia adelante en la red LSTM.
        """
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hidden

    def predict(self, char, hidden=None, device=torch.device('cpu'), top_k=None):
        """
        Predice el siguiente carácter dado un carácter de entrada.
        """
        with torch.no_grad():
            self.to(device)
            try:
                x = np.array([[self.char2int[char]]])
            except KeyError:
                return '', hidden

            x = self.one_hot_encode(x, len(self.chars))
            inputs = torch.from_numpy(x).to(device)

            out, hidden = self.forward(inputs, hidden)
            p = F.softmax(out, dim=2).data.to('cpu')

            if top_k is None:
                top_ch = np.arange(len(self.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()

            if top_k == 1:
                char = int(top_ch)
            else:
                p = p.numpy().squeeze()
                char = np.random.choice(top_ch, p=p / p.sum())

            return self.int2char[char], hidden

    def one_hot_encode(self, arr, n_labels):
        """
        Convierte una matriz en formato one-hot.
        """
        one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
        one_hot = one_hot.reshape((*arr.shape, n_labels))
        return one_hot

def save_checkpoint(net, opt, filename, train_history={}):
    """
    Guarda un modelo entrenado en un archivo.
    """
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict(),
                  'tokens': net.chars,
                  'train_history': train_history}

    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)

def load_checkpoint(filename):
    """
    Carga un modelo previamente guardado.
    """
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])

    return net, checkpoint