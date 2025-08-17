import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow import keras

class DiseasePredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Disease Prediction")
        self.root.geometry("650x800")
        self.root.configure(bg="#f0f4f7")

        # Load model
        model_path = r"D:\disease_prediction\sample_dataset\skin_cancer_model.h5"
        self.model = keras.models.load_model(model_path, compile=False)
        self.class_labels = ['basal_cell_carcinoma', 'melanoma', 'nevus']

        self.img = None

        self.build_ui()

    def build_ui(self):
        # Title label
        title = tk.Label(self.root, text="Skin Disease Prediction System", font=("Helvetica", 20, "bold"), bg="#f0f4f7")
        title.pack(pady=15)

        # Frame for symptoms with padding and border
        symptom_frame = tk.LabelFrame(self.root, text="Select Symptoms", padx=15, pady=15, font=("Helvetica", 14), bg="white")
        symptom_frame.pack(padx=20, pady=10, fill="x")

        self.symptoms = {
            "Redness or inflammation": tk.IntVar(),
            "Bleeding or oozing": tk.IntVar(),
            "Rough or scaly skin": tk.IntVar(),
            "Itching or irritation": tk.IntVar(),
            "Dark irregular spot": tk.IntVar(),
            "Wart-like growth": tk.IntVar(),
            "Slow-growing bump": tk.IntVar()
        }

        for symptom, var in self.symptoms.items():
            cb = tk.Checkbutton(symptom_frame, text=symptom, variable=var, font=("Helvetica", 12), bg="white")
            cb.pack(anchor="w", pady=3)

        # Frame for image upload and display
        image_frame = tk.Frame(self.root, bg="white", bd=2, relief="groove")
        image_frame.pack(padx=20, pady=15)

        self.img_label = tk.Label(image_frame, text="No Image Loaded", width=30, height=15, bg="#e1e5ea", fg="#666666", font=("Helvetica", 12))
        self.img_label.pack(padx=10, pady=10)

        btn_load = tk.Button(self.root, text="Upload Image", command=self.load_image, bg="#4a90e2", fg="white", font=("Helvetica", 14), width=20, relief="raised")
        btn_load.pack(pady=10)

        btn_predict = tk.Button(self.root, text="Predict Disease", command=self.predict, bg="#27ae60", fg="white", font=("Helvetica", 14), width=20, relief="raised")
        btn_predict.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 14), bg="#f0f4f7", fg="#333333", wraplength=600, justify="left")
        self.result_label.pack(pady=20, padx=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.img = Image.open(file_path).resize((224, 224))
            img_tk = ImageTk.PhotoImage(self.img)
            self.img_label.configure(image=img_tk, text="")
            self.img_label.image = img_tk
            self.result_label.config(text="")

    def predict(self):
        selected_symptoms = [s for s, var in self.symptoms.items() if var.get() == 1]
        if not selected_symptoms:
            messagebox.showwarning("Input Error", "Please select at least one symptom.")
            return
        if self.img is None:
            messagebox.showwarning("Input Error", "Please upload an image.")
            return

        img_array = np.array(self.img) / 255.0
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[..., :3]
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        predicted_class = self.class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        symptom_msg = "Symptoms selected: " + ", ".join(selected_symptoms)

        result_text = (f"Image Prediction: {predicted_class} (Confidence: {confidence:.4f})\n\n"
                       f"{symptom_msg}")

        self.result_label.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictionGUI(root)
    root.mainloop()

