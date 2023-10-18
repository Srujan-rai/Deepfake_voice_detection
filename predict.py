from tensorflow.keras.preprocessing import image
import numpy as np

from tensorflow import keras

image_height=640
image_width=480
model = keras.models.load_model('deepfake_model.h5')

new_image_path = ''
img = image.load_img(new_image_path, target_size=(image_height, image_width))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0


predictions = model.predict(img)


class_names = ['fake', 'real']
predicted_class_index = np.argmax(predictions)
predicted_class = class_names[predicted_class_index]

print(f"The image is classified as: {predicted_class}")
