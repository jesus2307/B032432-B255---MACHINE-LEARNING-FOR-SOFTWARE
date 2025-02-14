# B032432-B255---MACHINE-LEARNING-FOR-SOFTWARE
Descripción del Proyecto
Este proyecto implementa un modelo de autocompletado de código basado en redes neuronales recurrentes (RNN), específicamente utilizando LSTM bidireccionales. Se entrena con fragmentos de código y puede predecir la continuación más probable de una secuencia dada.

Estructura y Funcionamiento
1️⃣ Preprocesamiento de datos (preprocessing.py)
Carga el código fuente desde data/datos.txt.
Tokeniza el código, transformándolo en una secuencia de tokens.
Construye un vocabulario donde cada token recibe un índice.
Genera secuencias de entrenamiento de 30 tokens de longitud.
Guarda los datos procesados en processed_data.txt (no en processed_data.pt).
📌 Nota: Aunque preprocessing.py menciona processed_data.pt, este archivo no se usa en el flujo real del proyecto.

2️⃣ Entrenamiento del Modelo (train.py)
Carga los datos preprocesados desde processed_data.txt.
Define y entrena una red LSTM bidireccional (CharRNN):
Entrada: Secuencias de tokens codificadas en one-hot.
Arquitectura:
LSTM bidireccional con n_layers=3
Capa dropout para evitar sobreajuste.
Capa fully connected que predice el siguiente token.
Función de pérdida: CrossEntropyLoss()
Optimizador: Adam
Guarda el mejor modelo entrenado en models/training-datos.pth.
📌 Nota: Aquí se usa processed_data.txt, no processed_data.pt.

3️⃣ Evaluación del Modelo (evaluate.py)
Carga el modelo entrenado desde models/training-datos.pth.
Realiza predicciones sobre datos de validación.
Calcula métricas de rendimiento, como pérdida y precisión.
📌 Herramientas usadas: torch.nn.CrossEntropyLoss, torch.argmax para evaluar precisión.

4️⃣ Autocompletado de Código (autocomplete.py)
Carga el modelo entrenado (training-datos.pth) y el vocabulario.
Recibe un texto inicial y genera una continuación usando el modelo.
Usa una estrategia de muestreo con temperatura para controlar la creatividad de las predicciones.
Ejemplo de ejecución:
bash
Copiar
Editar
python src/autocomplete.py --input "def suma(a, b):"
Salida esperada (ejemplo):
python
Copiar
Editar
def suma(a, b):
    return a + b
📌 Nota: autocomplete.py usa CharRNN y softmax para generar código.

Arquitectura del Modelo (model.py)
El modelo CharRNN está basado en una LSTM bidireccional, que permite que el contexto fluya en ambas direcciones para mejorar la predicción.

🔹 Capas clave:

LSTM bidireccional (nn.LSTM) con múltiples capas.
Capa dropout (nn.Dropout) para reducir sobreajuste.
Capa fully connected (nn.Linear) que convierte la salida de LSTM en un token predecido.
🔹 Entradas y Salidas:

Entrada: Secuencia de tokens codificados en one-hot.
Salida: Distribución de probabilidad sobre tokens posibles.
Conclusión
Este proyecto es un sistema de autocompletado de código basado en IA, usando redes neuronales recurrentes (LSTM) entrenadas con código fuente. Su flujo de trabajo correcto es:

1️⃣ Preprocesar el código (processed_data.txt). 2️⃣ Entrenar la red LSTM (training-datos.pth). 3️⃣ Evaluar la precisión del modelo. 4️⃣ Usar el autocompletado en autocomplete.py.
