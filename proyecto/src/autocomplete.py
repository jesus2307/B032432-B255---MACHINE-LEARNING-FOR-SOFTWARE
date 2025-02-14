import torch
import torch.nn.functional as F
import numpy as np
import re
from model import CharRNN

def load_checkpoint(filepath):
    """ Carga el modelo desde un checkpoint y devuelve el modelo y la lista de tokens. """
    try:
        print(f"\n📌 Cargando modelo desde: {filepath}")
        with open(filepath, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        
        model = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print(f"✅ Modelo cargado con éxito ({len(checkpoint['tokens'])} tokens únicos)\n")
        return model, checkpoint['tokens']
    
    except FileNotFoundError:
        print(f"❌ Error: El archivo de checkpoint '{filepath}' no existe.")
        exit(1)
    except Exception as e:
        print(f"❌ Error al cargar el checkpoint: {e}")
        exit(1)

def generate_text(model, start_text, length=100, temperature=0.8):
    """ Genera texto a partir de un modelo CharRNN. """
    model.eval()
    with torch.no_grad():
        chars = [c for c in start_text if c in model.char2int]  # Filtrar caracteres desconocidos
        
        if not chars:
            print("❌ Error: La entrada no contiene caracteres válidos en el modelo.")
            return ""

        hidden = model.init_hidden(1)
        generated_chars = chars.copy()

        for _ in range(length):
            x = torch.tensor([[model.char2int[c] for c in chars]], dtype=torch.long)
            x = F.one_hot(x, num_classes=len(model.chars)).float()

            output, hidden = model(x, hidden)
            probs = F.softmax(output[:, -1, :] / max(temperature, 1e-5), dim=-1).squeeze()

            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = model.int2char.get(next_char_idx, '?')  # Manejo de errores
            generated_chars.append(next_char)

        generated_text = ''.join(generated_chars)
        generated_text = re.sub(r'(.)\1{5,}', r'\1\1', generated_text)  # Limita repeticiones

        return generated_text

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="🔠 Generador de texto con CharRNN.")
    parser.add_argument("--input", type=str, required=True, help="Texto inicial para la generación")
    parser.add_argument("--length", type=int, default=100, help="Longitud del texto generado")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura del muestreo (0.1 - 1.0)")
    parser.add_argument("--checkpoint", type=str, default="models/training-datos.pth")
    args = parser.parse_args()

    print("\n🚀 **Inicio de generación de texto** 🚀")
    print("="*50)

    net, tokens = load_checkpoint(args.checkpoint)
    
    if not args.input.strip():
        print("❌ Error: Se requiere un texto de entrada válido.")
        exit(1)

    generated_text = generate_text(net, args.input, args.length, args.temperature)

    print("\n📝 **Texto generado:**")
    print("="*50)
    print(generated_text)
    print("="*50)

