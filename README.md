  Disease Prediction System  

This project is a **Machine Learning-based Disease Prediction System** that predicts diseases based on symptoms and medical images.  
It includes a **Tkinter GUI** for easy interaction and supports both **symptom-based** and **image-based** prediction.  

  Features  
-  Predict diseases from **symptom checklist**  
-  Predict diseases from **medical images** using a trained CNN model  
-  User-friendly **GUI interface** (Tkinter)  
-  Modular code structure for easy extension  
-  `.gitignore` included (large model files & dataset excluded from repo)  


  Project Structure  
disease_prediction/
│
├── disease_gui.py # Tkinter GUI for prediction
├── disease_gui_improved.py # Enhanced version of GUI
├── disease_gui_pyqt.py # PyQt GUI version (optional)
├── predictor.py # ML prediction logic
├── train_model.py # Training script for models
├── check_counts.py # Utility script
├── .gitignore # Ignore large files (models, datasets)
└── sample_dataset/ # Example dataset (ignored in Git)

