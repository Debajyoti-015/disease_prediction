import sys
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, 
    QCheckBox, QGroupBox, QGridLayout, QMessageBox, QTextEdit, QSizePolicy
)
from PyQt6.QtGui import QPixmap
from tensorflow import keras

class DiseasePredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skin Disease Prediction (PyQt6)")
        self.setGeometry(100, 100, 600, 700)

        # Load model
        model_path = r"D:\disease_prediction\sample_dataset\skin_cancer_model.h5"
        self.model = keras.models.load_model(model_path, compile=False)
        self.class_labels = ['basal_cell_carcinoma', 'melanoma', 'nevus']

        self.image = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Symptom checklist group
        self.symptoms = {
            "Redness or inflammation": QCheckBox("Redness or inflammation"),
            "Bleeding or oozing": QCheckBox("Bleeding or oozing"),
            "Rough or scaly skin": QCheckBox("Rough or scaly skin"),
            "Itching or irritation": QCheckBox("Itching or irritation"),
            "Dark irregular spot": QCheckBox("Dark irregular spot"),
            "Wart-like growth": QCheckBox("Wart-like growth"),
            "Slow-growing bump": QCheckBox("Slow-growing bump")
        }

        symptom_group = QGroupBox("Select Symptoms")
        symptom_layout = QGridLayout()
        for i, checkbox in enumerate(self.symptoms.values()):
            symptom_layout.addWidget(checkbox, i // 2, i % 2)
        symptom_group.setLayout(symptom_layout)
        layout.addWidget(symptom_group)

        # Image preview label
        self.img_label = QLabel("No Image Loaded")
        self.img_label.setFixedSize(224, 224)
        self.img_label.setStyleSheet("border: 1px solid black;")
        self.img_label.setScaledContents(True)
        layout.addWidget(self.img_label)

        # Buttons
        btn_upload = QPushButton("Upload Image")
        btn_upload.clicked.connect(self.upload_image)
        layout.addWidget(btn_upload)

        btn_predict = QPushButton("Predict Disease")
        btn_predict.clicked.connect(self.predict)
        layout.addWidget(btn_predict)

        # Result display
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFixedHeight(100)
        layout.addWidget(self.result_display)

        self.setLayout(layout)

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image = Image.open(file_path).resize((224, 224))
            pixmap = QPixmap(file_path)
            self.img_label.setPixmap(pixmap.scaled(224, 224))
            self.result_display.clear()

    def predict(self):
        if self.image is None:
            QMessageBox.warning(self, "Input Error", "Please upload an image.")
            return

        selected_symptoms = [s for s, cb in self.symptoms.items() if cb.isChecked()]
        if not selected_symptoms:
            QMessageBox.warning(self, "Input Error", "Please select at least one symptom.")
            return

        # Prepare image for model
        img_array = np.array(self.image) / 255.0
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[..., :3]
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        predicted_class = self.class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Prepare symptom summary
        symptom_text = ", ".join(selected_symptoms)

        # Show results
        result_text = (f"Image Prediction: {predicted_class} (Confidence: {confidence:.4f})\n\n"
                       f"Selected Symptoms: {symptom_text}")

        self.result_display.setText(result_text)

def main():
    app = QApplication(sys.argv)
    window = DiseasePredictionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
