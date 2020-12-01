import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from tensorflow.keras import layers, models

class_names = ['Plane', 'car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('image_classifier.model')

path = glob.glob("C:/Users/44745/PycharmProjects/pythonProject7/*.jpg")
cv_img = []
for img in path:
    n = cv.imread(img)
    cv_img.append(n)

index = [0]*len(cv_img)
for i in range(len(cv_img)):
    cv_img[i] = cv.cvtColor(cv_img[i], cv.COLOR_BGR2RGB)
    prediction = model.predict(np.array([cv_img[i]]) / 255)
    index[i] = np.argmax(prediction)
    plt.subplot(int(round(len(cv_img)/4)), 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[index[i]])
    plt.imshow(cv_img[i], cmap=plt.cm.binary)

plt.show()

