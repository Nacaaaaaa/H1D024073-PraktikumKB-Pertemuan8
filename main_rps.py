import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Menetapkan jalur direktori data gambar
dataset_path = "./rockpaperscissors"

# 2. Inisialisasi generator data dan konfigurasi pembagian validasi sebesar 20%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# 3. Pembuatan aliran data otomatis untuk subset pelatihan
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

# 4. Pembuatan aliran data otomatis untuk subset validasi
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

# 5. Konstruksi arsitektur Convolutional Neural Network
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# 6. Tampilan parameter arsitektur CNN
model.summary()

# 7. Proses kompilasi parameter kalkulasi model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 8. Pelaksanaan proses training model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# 9. Pengujian kuantitatif performa model
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Hasil Evaluasi Validasi - Loss: {val_loss}, Akurasi: {val_acc}")

# 10. Ekstraksi probabilitas hasil prediksi akhir
predictions = model.predict(validation_generator)
print("Matriks Distribusi Probabilitas Prediksi:")
print(predictions)
