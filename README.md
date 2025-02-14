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


**Informe del Proyecto de Autocompletado de Código con IA**

## Introducción y Objetivos
El objetivo de este proyecto es desarrollar un modelo de inteligencia artificial capaz de realizar autocompletado de código en Python. Se ha implementado utilizando la biblioteca PyTorch y técnicas de procesamiento de lenguaje natural para entrenar una red neuronal que pueda predecir el siguiente token o línea de código a partir de un fragmento dado.

## Metodología

### Datos
Se han utilizado conjuntos de datos de código fuente en Python, los cuales han sido preprocesados y tokenizados para su uso en el modelo. Los datos están organizados en los siguientes archivos:
- `data/datos.txt`: Datos en crudo.
- `data/processed/processed_data.txt`: Datos procesados listos para el entrenamiento.

### Modelo
El modelo de autocompletado se basa en una arquitectura de red neuronal con PyTorch. Se han explorado tanto modelos secuenciales simples como arquitecturas basadas en transformadores para mejorar la precisión de las predicciones.
- `src/model.py`: Contiene la definición del modelo.
- `models/training-datos.pth`: Contiene el modelo entrenado.

### Entrenamiento
El entrenamiento del modelo se ha realizado con los datos preprocesados, utilizando optimización por descenso de gradiente y técnicas de regularización para evitar sobreajuste.
- `src/train.py`: Contiene el script de entrenamiento.
- `src/preprocessing.py`: Se encarga de la tokenización y preprocesamiento de datos.

### Evaluación
El modelo ha sido evaluado utilizando métricas estándar de predicción de secuencias, incluyendo precisión y pérdida.
- `src/evaluate.py`: Script para evaluar el rendimiento del modelo.

## Resultados y Discusión
El modelo entrenado ha demostrado ser capaz de predecir tokens de código con una precisión razonable, especialmente en estructuras sintácticas comunes de Python. Sin embargo, presenta dificultades con fragmentos complejos y dependencias a largo plazo.

Se ha identificado que el uso de modelos más avanzados basados en transformadores podría mejorar los resultados, aunque a costa de mayores requerimientos computacionales.

## Instrucciones de Ejecución
Para ejecutar el proyecto, siga los siguientes pasos:

1. **Instalar dependencias:**
   ```bash
   pip install torch tokenizers
   ```

2. **Ejecutar el entrenamiento:**
   ```bash
   python src/train.py
   ```

3. **Evaluar el modelo:**
   ```bash
   python src/evaluate.py
   ```

4. **Realizar inferencia para autocompletar código:**
   ```bash
   python src/autocomplete.py --input "def my_function("
   ```

## Conclusión
Este proyecto demuestra el potencial del aprendizaje automático para la predicción de código, aunque existen limitaciones en la precisión del modelo. Se recomienda continuar explorando modelos más sofisticados y aumentar el tamaño del conjunto de datos para mejorar los resultados.

## Referencias
- PyTorch Documentation: https://pytorch.org/docs/
- Técnicas de NLP para modelos de lenguaje.


