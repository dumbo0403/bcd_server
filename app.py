from flask import Flask, request, jsonify
import os
import base64
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 224

# Define the model architecture
model1 = Sequential()
model1.add(Conv2D(128, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(128, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(128, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(128))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))

model1.add(Dense(128))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))

model1.add(Dense(4))
model1.add(Activation('softmax'))

# Load the weights into the model
model1.load_weights(r'CNN_model.h5')

def prepare(image_array):
    new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

def base64_to_image(b64_string):
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def predict(image_array):
    processed_image = prepare(image_array)
    prediction = model1.predict(processed_image)
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability
    probability = prediction[0][predicted_class]  # Get the highest probability
    return predicted_class, probability

def process_images_in_b64(b64_string):
    image_array = base64_to_image(b64_string)
    probability = predict(image_array)
    return probability[1]

app = Flask(__name__)

@app.route('/submit', methods=['POST'])
def submit():

    data = request.get_json()    
    result = process_images_in_b64(data['image'])
    return jsonify(str(result))

if __name__ == '__main__':
    app.run(debug=True)