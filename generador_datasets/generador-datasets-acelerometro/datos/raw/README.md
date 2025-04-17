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
        x = np.array(hf['train']['scalograms'])[:num_samples, :num_scales, :]
        y = np.array(hf['train']['labels'])[:num_samples]

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
file_path = '/content/drive/MyDrive/Tesis/Accelerometer_Dataset/accelerometer_BR_256NS_20Scales_cwt_dataset.h5'

# Almacenar resultados
results = {}

# Probar cada configuración
for num_samples, num_scales in configurations:
    print(f"Probando con {num_samples} muestras y {num_scales} escalas...")
    
    # Cargar y normalizar el dataset
    x, y, num_classes = load_and_normalize_dataset(file_path, num_samples, num_scales)

    # Separar en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    # Aquí puedes llamar a la función de entrenamiento del modelo que ya tienes en modelo_CNN_basic_1vdataset.ipynb
    # Por ejemplo:
    # history = train_model(x_train, y_train, x_test, y_test, num_classes)

    # Almacenar resultados (ejemplo ficticio)
    # results[(num_samples, num_scales)] = history.history['val_accuracy'][-1]

# Mostrar resultados
# for config, accuracy in results.items():
#     print(f"Configuración {config}: Precisión de validación = {accuracy:.4f}")
```

### Descripción del Código:

1. **Importación de Bibliotecas**: Se importan las bibliotecas necesarias para manejar datos, realizar cálculos y crear gráficos.

2. **Función `load_and_normalize_dataset`**: Esta función carga el dataset desde un archivo HDF5, normaliza los datos entre 0 y 1 y convierte las etiquetas a formato one-hot.

3. **Configuraciones a Probar**: Se definen las configuraciones de número de muestras y escalas que se desean probar.

4. **Ciclo para Probar Configuraciones**: Para cada configuración, se carga y normaliza el dataset, se separa en conjuntos de entrenamiento y prueba, y se entrena el modelo (la función de entrenamiento del modelo debe ser definida en el archivo `modelo_CNN_basic_1vdataset.ipynb`).

5. **Almacenamiento y Visualización de Resultados**: Se almacenan los resultados de precisión de validación para cada configuración y se imprimen al final.

### Notas:
- Asegúrate de que la función de entrenamiento del modelo esté correctamente definida y sea accesible desde esta libreta.
- Puedes ajustar el código para que se adapte a tus necesidades específicas, como la visualización de resultados o el manejo de excepciones.
- Recuerda que necesitarás tener acceso al archivo HDF5 que contiene los datos del acelerómetro.