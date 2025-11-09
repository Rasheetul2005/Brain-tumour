# ===============================
# 🧠 Brain Tumor MRI Classification
# Google Colab Training Notebook
# ===============================

# Step 1️⃣ - Install dependencies
!pip install tensorflow pandas numpy matplotlib scikit-learn pillow tqdm

# Step 2️⃣ - Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt

print("✅ TensorFlow version:", tf.__version__)

# Step 3️⃣ - Mount Google Drive (optional: to load/save datasets/models)
from google.colab import drive
drive.mount('/content/drive')

# Change directory (optional) — store everything inside Drive
project_dir = "/content/drive/MyDrive/brain_tumor_mri_classification"
os.makedirs(project_dir, exist_ok=True)
os.chdir(project_dir)
print("📂 Project directory:", os.getcwd())

# Step 4️⃣ - Dataset setup
# Structure expected:
# data/train/glioma/, meningioma/, pituitary/, no_tumor/
# data/val/glioma/, meningioma/, pituitary/, no_tumor/
# You can upload or unzip your dataset here.
# Example: use Kaggle dataset (optional if you already have it uploaded)

# Example command to unzip a dataset (if uploaded to Drive)
# !unzip "/content/drive/MyDrive/dataset.zip" -d "./data"

# Step 5️⃣ - Data loading
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
img_size = (224, 224)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size, label_mode='int'
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=batch_size, label_mode='int'
)

class_names = train_ds.class_names
print("🧩 Classes found:", class_names)

# Step 6️⃣ - Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Step 7️⃣ - Model building (Transfer Learning using EfficientNetB0)
def build_transfer_model(num_classes, input_shape=(224,224,3)):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(base.input, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(class_names)
model = build_transfer_model(num_classes)
model.summary()

# Step 8️⃣ - Training
os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint("models/best_model.h5", monitor="val_accuracy", save_best_only=True, mode="max")
early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[checkpoint, early_stop])

# Step 9️⃣ - Fine-tuning (optional)
model.layers[1].trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=[checkpoint, early_stop])

# Step 🔟 - Save model
model.save("models/best_model.h5")
print("✅ Model saved at models/best_model.h5")

# Step 11️⃣ - Plot training performance
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.show()

# Step 12️⃣ - Test prediction on one image
import PIL
from PIL import Image

def preprocess_image(image_path, target_size=(224,224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    arr = np.array(image)/255.0
    return np.expand_dims(arr, 0)

# Example:
# test_image_path = "data/val/glioma/image(1).jpg"
# image = preprocess_image(test_image_path)
# preds = model.predict(image)[0]
# idx = int(preds.argmax())
# conf = float(preds[idx])
# print(f"🧠 Prediction: {class_names[idx]}  (confidence: {conf:.3f})")
