import os
import re
import unidecode

# Definir directorios de datos
data_dir = "data/processed/"
data_file = "data/datos.txt"

# Crear el directorio de datos si no existe
os.makedirs(data_dir, exist_ok=True)

def clean_content(x):
    """
    Función de preprocesamiento para limpiar el contenido del archivo.
    - Elimina comentarios
    - Reduce múltiples saltos de línea
    - Normaliza caracteres especiales
    
    Args:
        x (str): Texto de entrada.
    
    Returns:
        str: Texto preprocesado.
    """
    x = unidecode.unidecode(x)  # Elimina acentos y caracteres especiales
    x = re.sub(r'#.*$', '', x, flags=re.MULTILINE)  # Elimina comentarios en línea
    x = re.sub(r'^\s*\n', '', x, flags=re.MULTILINE)  # Elimina líneas vacías o con solo espacios/tabulaciones
    x = re.sub(r'\n{2,}', '\n', x, flags=re.MULTILINE)  # Reduce múltiples saltos de línea a uno
    x = "\n".join(line.strip() for line in x.split("\n"))  # Elimina espacios al inicio y final de cada línea
    return x

# Leer y procesar datos desde el archivo si existe
if os.path.exists(data_file):
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = f.read()
        cleaned_data = clean_content(raw_data)

        # Guardar datos preprocesados
        processed_file = os.path.join(data_dir, "processed_data.txt")
        with open(processed_file, "w", encoding="utf-8") as f:
            f.write(cleaned_data)
        print(f" Datos preprocesados guardados en {processed_file}")

    except Exception as e:
        print(f" Error al procesar el archivo: {e}")

else:
    print(f" El archivo {data_file} no existe.")
