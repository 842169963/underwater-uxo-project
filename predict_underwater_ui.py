import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk

from predict_underwater import load_model, predict_image


class UnderwaterPredictUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Underwater UXO Predictor")
        self.root.geometry("900x620")

        self.model = None
        self.model_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.result_text = tk.StringVar(value="Prediction: -")
        self.conf_text = tk.StringVar(value="Confidence: -")

        self.preview_label = None
        self.preview_photo = None

        self._build()

    def _build(self):
        top = tk.Frame(self.root, padx=12, pady=12)
        top.pack(fill="x")

        tk.Label(top, text="Model (.pth):").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.model_path, width=90).grid(row=1, column=0, padx=(0, 8), sticky="w")
        tk.Button(top, text="Browse Model", command=self.choose_model).grid(row=1, column=1, sticky="w")
        tk.Button(top, text="Load Model", command=self.load_selected_model).grid(row=1, column=2, padx=(8, 0), sticky="w")

        tk.Label(top, text="Image:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        tk.Entry(top, textvariable=self.image_path, width=90).grid(row=3, column=0, padx=(0, 8), sticky="w")
        tk.Button(top, text="Browse Image", command=self.choose_image).grid(row=3, column=1, sticky="w")
        tk.Button(top, text="Predict", command=self.predict_current_image).grid(row=3, column=2, padx=(8, 0), sticky="w")

        status = tk.Frame(self.root, padx=12, pady=8)
        status.pack(fill="x")
        tk.Label(status, textvariable=self.result_text, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(status, textvariable=self.conf_text, font=("Segoe UI", 11)).pack(anchor="w")

        preview_frame = tk.Frame(self.root, padx=12, pady=12)
        preview_frame.pack(fill="both", expand=True)
        self.preview_label = tk.Label(preview_frame, text="Image preview will appear here")
        self.preview_label.pack(anchor="center", expand=True)

    def choose_model(self):
        path = filedialog.askopenfilename(
            title="Choose model file",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self.model_path.set(path)

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_path.set(path)
            self.show_preview(path)

    def load_selected_model(self):
        model_path = self.model_path.get().strip()
        if not model_path:
            messagebox.showwarning("Missing model", "Please select a model file first.")
            return
        if not Path(model_path).exists():
            messagebox.showerror("File not found", f"Model not found:\n{model_path}")
            return
        try:
            self.model = load_model(model_path)
            messagebox.showinfo("Model loaded", f"Loaded model:\n{model_path}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def show_preview(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((760, 420))
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_photo, text="")
        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    def predict_current_image(self):
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Please load a model first.")
            return
        image_path = self.image_path.get().strip()
        if not image_path:
            messagebox.showwarning("Missing image", "Please select an image first.")
            return
        if not Path(image_path).exists():
            messagebox.showerror("File not found", f"Image not found:\n{image_path}")
            return

        try:
            label, conf, probs = predict_image(self.model, image_path)
            self.result_text.set(f"Prediction: {label}")
            self.conf_text.set(
                f"Confidence: {conf:.4f} | P(UXO)={probs[0]:.4f}, P(non_UXO)={probs[1]:.4f}"
            )
        except Exception as e:
            messagebox.showerror("Prediction failed", str(e))


def main():
    root = tk.Tk()
    UnderwaterPredictUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
