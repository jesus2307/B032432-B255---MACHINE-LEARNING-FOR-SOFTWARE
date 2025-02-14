import os
import tokenize
import torch
from collections import Counter

# Configuración
SEQ_LENGTH = 30  # Número de tokens usados como entrada
DATA_PATH = "data/datos.txt"  # Ruta del archivo con código fuente
PROCESSED_PATH = "data/processed/processed_data.pt"  # Donde se guardarán los datos preprocesados

# Función para tokenizar código

def tokenize_code(code):
    tokens = []
    try:
        tokens = [tok.string for tok in tokenize.generate_tokens(iter(code.splitlines()).__next__)]
    except tokenize.TokenError:
        pass
    return tokens

# Leer el archivo con código
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = f.read()

    # Tokenizar el código
    tokens = tokenize_code(raw_data)

    # Construcción del vocabulario
    counter = Counter(tokens)
    vocab = {token: i for i, (token, _) in enumerate(counter.most_common())}
    inv_vocab = {i: token for token, i in vocab.items()}

    # Crear secuencias de entrenamiento
    input_data, target_data = [], []
    for i in range(len(tokens) - SEQ_LENGTH):
        input_seq = tokens[i:i + SEQ_LENGTH]
        target_token = tokens[i + SEQ_LENGTH]
        input_data.append([vocab[tok] for tok in input_seq if tok in vocab])
        target_data.append(vocab.get(target_token, 0))

    # Convertir a tensores
    input_tensor = torch.tensor(input_data, dtype=torch.long)
    target_tensor = torch.tensor(target_data, dtype=torch.long)

    # Guardar datos preprocesados
    torch.save((input_tensor, target_tensor, vocab, inv_vocab), PROCESSED_PATH)
    print(f"Datos preprocesados guardados en {PROCESSED_PATH}")
else:
    print(f"El archivo {DATA_PATH} no existe.")

# Función para cargar los datos preprocesados

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file {filepath} does not exist.")
    
    # Cargar los datos con torch.load()
    data = torch.load(filepath, map_location=torch.device('cpu'))

    # Extraer los datos correctamente
    input_tensor, target_tensor, vocab, inv_vocab = data
    
    chars = list(vocab.keys())
    char2int = vocab
    int2char = inv_vocab
    
    return chars, char2int, int2char, input_tensor

# Función para evaluar el modelo

def evaluate_model(model, data, char2int, n_steps=32):
    if model is None:
        print("Model not loaded. Exiting evaluation.")
        return

    print("Evaluating model...")
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        hidden = model.init_hidden(1)  # Batch size de 1 para evaluación
        for i in range(0, len(data) - n_steps, n_steps):
            x = data[i:i + n_steps]
            y = data[i + 1:i + n_steps + 1]

            # Convertir caracteres a índices en el vocabulario
            x = torch.tensor([char2int.get(c, 0) for c in x], dtype=torch.long).unsqueeze(0)
            y = torch.tensor([char2int.get(c, 0) for c in y], dtype=torch.long).unsqueeze(0)

            # One-hot encoding (asegurando que no haya índices fuera de rango)
            x = F.one_hot(x, num_classes=len(model.chars)).float()

            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, len(model.chars)), y.view(-1))
            total_loss += loss.item()

            pred = torch.argmax(output, dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    
    avg_loss = total_loss / max(1, (len(data) // n_steps))
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print("Model loaded successfully.")
    print(f"Evaluation complete. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
