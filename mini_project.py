import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
base_dir = 'C:/Users/admin/Desktop/tomato'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

print("Train data count:", len(os.listdir(train_dir)))
print("Validation data count:", len(os.listdir(val_dir)))

train_data_directory = pathlib.Path(train_dir)
val_data_directory = pathlib.Path(val_dir)

class_names = sorted(item.name for item in train_data_directory.glob('*'))
print("Class names:", class_names)

train_data_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_data_gen = ImageDataGenerator(rescale=1/255.)

train_data = train_data_gen.flow_from_directory(
    train_dir, 
    target_size=(224, 224), 
    batch_size=32,
    class_mode='categorical'
)

val_data = val_data_gen.flow_from_directory(
    val_dir, 
    target_size=(224, 224), 
    batch_size=32,
    class_mode='categorical'
)

# Model creation
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=val_data, epochs=3, verbose=1)
model.evaluate(train_data, verbose=1)
model.save("tomato.keras")

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
loaded_model = tf.keras.models.load_model('tomato.keras')
def preprocess_image(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_array / 255.0
    return img_preprocessed

img_path = "C:/Users/admin/Desktop/test.JPG"
img = preprocess_image(img_path)
prediction = loaded_model.predict(img)
predicted_class_index = np.argmax(prediction)
class_labels = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", 
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", 
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
] 
predicted_class_label = class_labels[predicted_class_index]

if predicted_class_label == "Tomato___healthy":
    health_status = "healthy"
else:
    health_status = "not healthy"

print("Predicted class:", predicted_class_label)
print("Health status:", health_status)