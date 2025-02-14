# B032432-MACHINE-LEARNING-FOR-SOFTWARE
# Descripción del Proyecto

Este proyecto implementa un **modelo de autocompletado de código basado en redes neuronales recurrentes (RNN)**, específicamente utilizando **LSTM bidireccionales**. Se entrena con fragmentos de código y puede predecir la continuación más probable de una secuencia dada.

---

#  Estructura y Funcionamiento

##  Preprocesamiento de datos (`preprocessing.py`)
- **Carga el código fuente** desde `data/datos.txt`.
- **Tokeniza el código**, transformándolo en una secuencia de tokens.
- **Construye un vocabulario** donde cada token recibe un índice.
- **Genera secuencias de entrenamiento** de 30 tokens de longitud.
- **Guarda los datos procesados en `processed_data.txt`


---

##  Entrenamiento del Modelo (`train.py`)
- **Carga los datos preprocesados desde `processed_data.txt`**.
- **Define y entrena una red LSTM bidireccional (`CharRNN`)**:
  - **Entrada:** Secuencias de tokens codificadas en one-hot.
  - **Arquitectura:**
    - **LSTM bidireccional** con `n_layers=3`
    - **Capa dropout** para evitar sobreajuste.
    - **Capa fully connected** que predice el siguiente token.
  - **Función de pérdida:** `CrossEntropyLoss()`
  - **Optimizador:** `Adam`
- **Guarda el mejor modelo entrenado en `models/training-datos.pth`**.

 **Nota:** Aquí se usa `processed_data.txt`, no `processed_data.pt`.

---

## Evaluación del Modelo (`evaluate.py`)
- **Carga el modelo entrenado desde `models/training-datos.pth`**.
- **Realiza predicciones sobre datos de validación**.
- **Calcula métricas de rendimiento**, como pérdida y precisión.

 **Herramientas usadas:** `torch.nn.CrossEntropyLoss`, `torch.argmax` para evaluar precisión.

---

##  Autocompletado de Código (`autocomplete.py`)
- **Carga el modelo entrenado (`training-datos.pth`) y el vocabulario.**
- **Recibe un texto inicial** y genera una continuación usando el modelo.
- **Usa una estrategia de muestreo con temperatura** para controlar la creatividad de las predicciones.

### **Ejemplo de ejecución**
```bash
python src/autocomplete.py --input "def suma(a, b):"

