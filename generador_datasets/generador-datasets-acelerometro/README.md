# Importar las bibliotecas necesarias
import h5py
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Función para cargar y normalizar el dataset
def load_and_normalize_dataset(file_path, num_samples, num_scales):
    with h5py.File(file_path, 'r') as hf:
        # Cargar escalogramas y etiquetas
        x = np.array(hf['train']['scalograms'][:num_samples, :num_scales, :])
        y = np.array(hf['train']['labels'][:num_samples])
        
        # Normalizar los datos
        max_value = np.max(x)
        x = x.astype('float32') / max_value
        
        # Convertir etiquetas a formato one-hot
        num_classes = len(np.unique(y))
        y = to_categorical(y, num_classes)
        
        return x, y, num_classes

# Configuraciones a probar
configurations = [
    (64, 100),
    (128, 50),
    (512, 13),
    (1024, 6)
]

# Ruta al archivo HDF5
file_path = '/path/to/your/accelerometer_dataset.h5'

# Almacenar resultados
results = {}

# Probar cada configuración
for num_samples, num_scales in configurations:
    print(f"Probando con {num_samples} muestras y {num_scales} escalas...")
    
    # Cargar y normalizar el dataset
    x, y, num_classes = load_and_normalize_dataset(file_path, num_samples, num_scales)
    
    # Separar en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    # Aquí puedes llamar a tu modelo de CNN y entrenarlo
    # tiny_cnn = create_tiny_cnn_model(input_shape=x_train.shape[1:], num_classes=num_classes)
    # tiny_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # history = tiny_cnn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64)
    
    # Evaluar el modelo
    # test_loss, test_acc = tiny_cnn.evaluate(x_test, y_test)
    # results[(num_samples, num_scales)] = test_acc
    # print(f"Precisión para {num_samples} muestras y {num_scales} escalas: {test_acc:.4f}")

# Mostrar resultados
# for config, accuracy in results.items():
#     print(f"Muestras: {config[0]}, Escalas: {config[1]} -> Precisión: {accuracy:.4f}")
```

### Descripción del Código:

1. **Importación de Bibliotecas**: Se importan las bibliotecas necesarias para manejar datos, realizar cálculos y construir modelos.

2. **Función `load_and_normalize_dataset`**: Esta función carga el dataset desde un archivo HDF5, normaliza los datos entre 0 y 1 y convierte las etiquetas a formato one-hot.

3. **Configuraciones a Probar**: Se definen las configuraciones de número de muestras y escalas que se desean probar.

4. **Ciclo para Probar Configuraciones**: Para cada configuración, se carga y normaliza el dataset, se separa en conjuntos de entrenamiento y prueba, y se entrena el modelo de CNN.

5. **Evaluación del Modelo**: Se evalúa el modelo en el conjunto de prueba y se almacenan los resultados.

6. **Mostrar Resultados**: Finalmente, se imprimen los resultados de precisión para cada configuración.

### Notas:
- Asegúrate de ajustar la ruta del archivo HDF5 (`file_path`) a la ubicación correcta de tu dataset.
- Descomenta las líneas relacionadas con el modelo de CNN y la evaluación para ejecutar el entrenamiento y la evaluación del modelo.
- Puedes personalizar el modelo y los hiperparámetros según tus necesidades.