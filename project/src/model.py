import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    """
    Implementación de una red neuronal recurrente basada en LSTM para la generación de texto carácter a carácter.
    """
    def __init__(self, tokens, n_hidden=1024, n_layers=4, drop_prob=0.3):
        """
        Inicializa el modelo CharRNN.
        
        Args:
            tokens (list): Lista de caracteres únicos en los datos de entrenamiento.
            n_hidden (int): Número de unidades ocultas en las capas LSTM.
            n_layers (int): Número de capas LSTM.
            drop_prob (float): Probabilidad de dropout para regularización.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))  # Mapeo de índice a carácter
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}  # Mapeo de carácter a índice
        
        # Capa LSTM bidireccional
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                             dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden * 2, len(self.chars))  # Capa de salida ajustada para LSTM bidireccional

    def forward(self, x, hidden):
        """
        Paso hacia adelante a través del modelo.
        
        Args:
            x (Tensor): Entrada a la red en formato de tensores.
            hidden (tuple): Estado oculto de la LSTM.
        
        Returns:
            Tensor: Salida del modelo.
            tuple: Nuevo estado oculto.
        """
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hidden

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        """
        Inicializa los estados ocultos de la red LSTM.
        
        Args:
            batch_size (int): Tamaño del batch.
            device (torch.device): Dispositivo de ejecución (CPU o GPU).
        
        Returns:
            tuple: Estados ocultos inicializados.
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers * 2, batch_size, self.n_hidden).zero_().to(device))
        return hidden

def save_checkpoint(net, opt, filename, train_history={}):
    """
    Guarda el modelo entrenado en un archivo.
    
    Args:
        net (CharRNN): Modelo entrenado.
        opt (torch.optim.Optimizer): Optimizador utilizado en el entrenamiento.
        filename (str): Ruta del archivo donde se guardará el modelo.
        train_history (dict): Historial del entrenamiento (opcional).
    """
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict(),
                  'tokens': net.chars,
                  'train_history': train_history}
    torch.save(checkpoint, filename)

def load_checkpoint(filename):
    """
    Carga un modelo previamente guardado desde un archivo.
    
    Args:
        filename (str): Ruta del archivo del modelo guardado.
    
    Returns:
        CharRNN: Modelo cargado.
        dict: Diccionario con el checkpoint.
    """
    checkpoint = torch.load(filename, map_location='cpu')
    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])
    return net, checkpoint

if __name__ == "__main__":
    print("Model definition is complete.")
