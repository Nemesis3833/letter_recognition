import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageGrab, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

model = load_model("letter_recognition_model.keras")

prediction_delay = 1000
prediction_job = None

def predict_canvas():
    try:
        canvas.update()
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1))
        img = img.convert("L")
        img = img.resize((32, 32))
        img = np.array(img, dtype=np.float32) / 255.0  
        img = 1 - img  
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_label = np.argmax(predictions, axis=1)[0]
        predicted_letter = chr(predicted_label + 65)
        result_label.config(text=f"Predicted Letter: {predicted_letter}")
        update_confidence_graph(predictions[0])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process canvas: {e}")

def update_confidence_graph(predictions):
    letters = [chr(i + 65) for i in range(26)]
    ax.clear()
    ax.bar(letters, predictions, color='skyblue')
    ax.set_xlabel("Letters")
    ax.set_ylabel("Confidence")
    ax.set_title("Model Confidence for Each Letter")
    ax.set_ylim(0, 1)
    graph_canvas.draw()

def clear_canvas():
    global prediction_job
    if prediction_job:
        root.after_cancel(prediction_job)
        prediction_job = None
    canvas.delete("all")
    result_label.config(text="Predicted Letter: ")
    ax.clear()
    ax.set_title("Model Confidence for Each Letter")
    graph_canvas.draw()

def draw(event):
    global prediction_job
    x, y = event.x, event.y
    canvas.create_oval(x-5, y-5, x+5, y+5, fill="black")
    if prediction_job:
        root.after_cancel(prediction_job)
    prediction_job = root.after(prediction_delay, predict_canvas)

root = tk.Tk()
root.title("Letter Recognition")

canvas = tk.Canvas(root, width=300, height=300, bg="white")
canvas.pack(pady=10)

canvas.bind("<B1-Motion>", draw)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack(pady=10)

result_label = tk.Label(root, text="Predicted Letter: ", font=("Arial", 16))
result_label.pack(pady=10)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Model Confidence for Each Letter")
ax.set_ylim(0, 1)
graph_canvas = FigureCanvasTkAgg(fig, master=root)
graph_canvas.get_tk_widget().pack(pady=10)

root.mainloop()