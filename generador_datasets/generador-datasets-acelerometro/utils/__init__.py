# Importar las librerías necesarias
import h5py
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
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

# Función para entrenar el modelo
def train_model(x_train, y_train, x_val, y_val, num_classes):
    input_shape = x_train.shape[1:]
    
    # Crear el modelo
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
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Entrenar el modelo
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32)
    
    return model, history

# Configuraciones a probar
configurations = [
    (64, 100),
    (128, 50),
    (512, 13),
    (1024, 6)
]

# Ruta al archivo HDF5
file_path = '/content/drive/MyDrive/Tesis/Accelerometer_Dataset/accelerometer_BR_256NS_20Scales_cwt_dataset.h5'

# Probar diferentes configuraciones
for num_samples, num_scales in configurations:
    print(f"Configuración: {num_samples} muestras, {num_scales} escalas")
    
    # Cargar y normalizar el dataset
    x, y, num_classes = load_and_normalize_dataset(file_path, num_samples, num_scales)
    
    # Separar en conjuntos de entrenamiento y validación
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    # Entrenar el modelo
    model, history = train_model(x_train, y_train, x_val, y_val, num_classes)
    
    # Evaluar el modelo
    val_loss, val_accuracy = model.evaluate(x_val, y_val)
    print(f"Precisión en validación: {val_accuracy:.4f}\n")
    
    # Graficar la precisión y la pérdida
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy for {num_samples} samples and {num_scales} scales')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss for {num_samples} samples and {num_scales} scales')
    plt.legend()
    
    plt.show()