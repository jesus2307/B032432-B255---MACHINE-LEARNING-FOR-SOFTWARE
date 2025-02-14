import torch
import torch.nn.functional as F
import numpy as np
from model import CharRNN

# Permitir la carga segura si usamos weights_only=True
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        model = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model, checkpoint['tokens']
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None

def evaluate_model(model, data, n_steps=32):
    if model is None:
        print("Model not loaded. Exiting evaluation.")
        return

    print("\nEvaluating model...\n")
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        hidden = model.init_hidden(1)
        for i in range(0, len(data) - n_steps, n_steps):
            x = data[i:i + n_steps]
            y = data[i + 1:i + n_steps + 1]

            x = torch.tensor([model.char2int.get(c, 0) for c in x], dtype=torch.long).unsqueeze(0)
            y = torch.tensor([model.char2int.get(c, 0) for c in y], dtype=torch.long).unsqueeze(0)

            x = torch.clamp(x, min=0, max=len(model.chars) - 1)
            y = torch.clamp(y, min=0, max=len(model.chars) - 1)

            x = F.one_hot(x, num_classes=len(model.chars)).float()

            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, len(model.chars)), y.view(-1))
            total_loss += loss.item()

            pred = torch.argmax(output, dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()

    avg_loss = total_loss / max(1, (len(data) // n_steps))
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    print("Model loaded successfully.\n")
    print(f"Evaluation complete. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    checkpoint_path = "models/training-datos.pth"
    data_path = "data/processed/processed_data.txt"
    
    model, chars = load_checkpoint(checkpoint_path)
    if chars:
        evaluate_model(model, chars)
