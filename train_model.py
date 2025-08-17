# train_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======== CONFIG ========
CSV_PATH = "sample_dataset/disease_data.csv"     # Path to your CSV dataset
IMAGE_DATASET_PATH = "sample_dataset/images"     # Folder containing disease images
CSV_MODEL_PATH = "model_csv.pkl"                 # Saved CSV model
IMG_MODEL_PATH = "model_img.pkl"                 # Saved Image CNN model
IMG_MODEL_H5 = "model_img.h5"                     # Saved Keras model

# ======== 1. TRAIN CSV-BASED MODEL ========
if os.path.exists(CSV_PATH):
    print("\n[1] Training CSV-based Disease Prediction Model...")

    df = pd.read_csv(CSV_PATH)

    # Assuming last column is the target
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]   

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_csv = RandomForestClassifier(n_estimators=100, random_state=42)
    model_csv.fit(X_train, y_train)

    y_pred = model_csv.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"CSV Model Accuracy: {acc:.2f}")

    joblib.dump(model_csv, CSV_MODEL_PATH)
    print(f"CSV model saved as {CSV_MODEL_PATH}")
else:
    print("CSV dataset not found! Skipping CSV model training.")

# ======== 2. TRAIN IMAGE-BASED CNN MODEL ========
if os.path.exists(IMAGE_DATASET_PATH):
    print("\n[2] Training Image-based Disease Recognition Model...")

    # Data Augmentation
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        IMAGE_DATASET_PATH,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        IMAGE_DATASET_PATH,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Build CNN model
    model_img = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_gen.num_classes, activation='softmax')
    ])

    model_img.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model_img.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5
    )

    # Save CNN model
    model_img.save(IMG_MODEL_H5)  
    joblib.dump(model_img, IMG_MODEL_PATH)  # Optional, but Keras models usually saved as .h5

    print(f"Image model saved as {IMG_MODEL_H5} and {IMG_MODEL_PATH}")
else:
    print("Image dataset folder not found! Skipping image model training.")

print("\n✅ Training complete for all available datasets.")
