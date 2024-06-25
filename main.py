import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import cv2
from PIL import ImageOps, Image 
import numpy as np


def preprocess(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_greyscale = image.convert('L')
    image_resize = image_greyscale.resize((28,28))
    image_array = np.array(image_resize)
    image_rescaled = image_array/255
    image_reshaped = np.expand_dims(image_rescaled,axis=(0,-1))    
    return image_reshaped
    
def prediction(image_array: np.ndarray) -> str:
    model = models.load_model('models/CNN_digit.h5')
    prob = model.predict(processed_image)
    result = np.argmax(prob)
    return result

if __name__=="__main__":
    processed_image = preprocess("3_mnist.png")
    result = prediction(processed_image)
    print(f'The predicted digit is {result}')