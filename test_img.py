import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("cat_or_human_model.keras")

IMG_SIZE = 224  # MUST match training

# Load image
img = image.load_img("test2.jpg", target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array, verbose=0)[0][0]

if prediction > 0.5:
    print(f"👤 HUMAN ({prediction:.2f})")
else:
    print(f"🐱 CAT ({1 - prediction:.2f})")
