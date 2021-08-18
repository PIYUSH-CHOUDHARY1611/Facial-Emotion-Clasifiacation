# import os
import cv2
# import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# from keras.preprocessing import image
model = load_model('')
model.load_weights('model7.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# a = input("enter image path ")
test_img = cv2.imread('0085.jpg', 0)
# display input image
plt.imshow(test_img, cmap='gray')
plt.show()

test_img = cv2.resize(test_img, (48, 48))

plt.imshow(test_img, cmap='gray')
plt.show()

#print(test_img.shape)


test_img = test_img.reshape(1, 48, 48, 1)

print(model.predict(test_img))


if model.predict(test_img)[0][0] == 1:
    print("Anger")
elif model.predict(test_img)[0][1] == 1:
    print("Neutral")
elif model.predict(test_img)[0][2] == 1:
    print("Fear")
elif model.predict(test_img)[0][3] == 1:
    print("Happy")
elif model.predict(test_img)[0][4] == 1:
    print("Sad")
else:
    print("Suprise")

# label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
