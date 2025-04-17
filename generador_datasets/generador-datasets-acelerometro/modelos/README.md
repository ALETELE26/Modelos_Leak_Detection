### Nueva Libreta de Jupyter: `generar_datasets_y_evaluar_modelos.ipynb`

```python
# Importar las librerías necesarias
import h5py
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import layers

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

# Función para crear y compilar el modelo
def create_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(8, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Configuraciones a probar
configurations = [
    (64, 100),
    (128, 50),
    (512, 13),
    (1024, 6)
]

# Ruta al archivo HDF5
file_path = '/content/drive/MyDrive/Tesis/Accelerometer_Dataset/accelerometer_BR_256NS_20Scales_cwt_dataset.h5'

# Evaluar cada configuración
for num_samples, num_scales in configurations:
    print(f"Evaluando configuración: {num_samples} muestras y {num_scales} escalas")
    
    # Cargar y normalizar el dataset
    x, y, num_classes = load_and_normalize_dataset(file_path, num_samples, num_scales)
    
    # Separar en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    # Adaptar dimensiones para CNN
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Crear y entrenar el modelo
    model = create_model(x_train.shape[1:], num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'best_model_{num_samples}_{num_scales}.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    # Entrenar el modelo
    history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=callbacks)
    
    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

    # Graficar resultados
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'Accuracy: {num_samples} muestras, {num_scales} escalas')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'Loss: {num_samples} muestras, {num_scales} escalas')
    plt.legend()
    plt.show()
```

### Descripción del Código

1. **Importación de Librerías**: Se importan las librerías necesarias para el manejo de datos, la creación de modelos y la visualización.

2. **Carga y Normalización**: La función `load_and_normalize_dataset` carga los datos desde un archivo HDF5, normaliza los escalogramas entre 0 y 1 y convierte las etiquetas a formato one-hot.

3. **Creación del Modelo**: La función `create_model` define un modelo CNN básico.

4. **Configuraciones**: Se definen las configuraciones de número de muestras y escalas a probar.

5. **Evaluación de Configuraciones**: Para cada configuración, se carga el dataset, se separa en conjuntos de entrenamiento y prueba, se adapta la forma de los datos, se crea y entrena el modelo, y finalmente se evalúa la precisión.

6. **Visualización**: Se grafican las métricas de precisión y pérdida durante el entrenamiento.

### Ejecución

Guarda el código anterior en una nueva libreta de Jupyter y ejecútalo en un entorno que tenga acceso a los datos del acelerómetro. Asegúrate de que el archivo HDF5 esté en la ruta correcta. Esto te permitirá evaluar qué configuración de muestras y escalas produce la mejor precisión en la clasificación.