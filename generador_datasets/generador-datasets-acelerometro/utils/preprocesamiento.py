# Importar las bibliotecas necesarias
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

# Función para entrenar y evaluar el modelo
def train_and_evaluate_model(x_train, y_train, x_test, y_test, num_classes):
    # Crear el modelo
    input_shape = x_train.shape[1:]
    model = create_tiny_cnn_model(input_shape, num_classes)

    # Compilar el modelo
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2)

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

    # Visualizar el proceso de entrenamiento
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.show()

# Configuraciones a probar
configurations = [
    (64, 100),
    (128, 50),
    (512, 13),
    (1024, 6)
]

# Ruta al archivo HDF5
file_path = '/content/drive/MyDrive/Tesis/Accelerometer_Dataset/accelerometer_BR_256NS_20Scales_cwt_dataset.h5'

# Probar cada configuración
for num_samples, num_scales in configurations:
    print(f"Probando con {num_samples} muestras y {num_scales} escalas...")
    
    # Cargar y normalizar el dataset
    x, y, num_classes = load_and_normalize_dataset(file_path, num_samples, num_scales)

    # Dividir el dataset en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Entrenar y evaluar el modelo
    train_and_evaluate_model(x_train, y_train, x_test, y_test, num_classes)