import numpy as np
import keras
import cv2


model = keras.models.load_model("./model.keras")
model.compile(loss="BinaryCrossentropy", optimizer="adam", metrics=["accuracy"])
classes = ["Artur", "Unknown"]


img_path = ""  # img_path
img = keras.preprocessing.image.load_img(img_path, target_size=(225, 225))
foto = cv2.imread(img_path)


img = keras.preprocessing.image.img_to_array(img)
img /= 255
img = np.expand_dims(img, axis=0)


prediction = model.predict(img)
result = int(np.amax(prediction))
print(result)
print(classes[result])
