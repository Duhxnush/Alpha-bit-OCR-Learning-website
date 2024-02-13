import cv2
import sys
import numpy as np
from keras import models

MODEL_PATH = "D:\\Alphabit website\\tf-cnn-model.h5"

def predict_digit(image_path):
    
    # load model
    model = models.load_model(MODEL_PATH)
    print("[INFO] Loaded model from disk.")

    image = cv2.imread(image_path, 0)      
    image1 = cv2.resize(image, (28,28))    # For cv2.imshow: dimensions should be 28x28
    image2 = image1.reshape(1,28,28,1)

    cv2.imshow('digit', image1 )
    pred = np.argmax(model.predict(image2), axis=-1)
    return pred[0]    

predicted_digit = predict_digit("D:\\Alphabit website\\temp.png")
print('Predicted Digit: {}'.format(predicted_digit))
 
