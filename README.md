  Disease Prediction System  

A Python-based application that predicts diseases using machine learning models trained on symptoms and medical images.  
It provides a Tkinter GUI for user interaction and stores relevant information in a database for future use.

Features
- User-friendly Tkinter GUI (`disease_gui_improved.py`)  
- Predict diseases from symptoms or uploaded medical images 
- Uses CNN model (H5) for image classification  
- Supports Pickle model (PKL) for symptom-based predictions  
- Training script included (`train_model.py`)  
- Dataset provided under `sample_dataset/`  

Project Structure
disease_prediction/
│
├── sample_dataset/ # Sample dataset for testing/training
├── disease_gui_improved.py # Tkinter-based GUI for predictions
├── train_model.py # Script to train ML models
├── model_img.h5 # Trained CNN model (image-based)
├── model_img.pkl # Trained ML model (pickle)
├── README.md # Documentation
└── .gitignore # Git ignore rules
