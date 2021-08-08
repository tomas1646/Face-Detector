import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Para imagenes:
img = cv2.imread('MC.jpg')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 3)


cv2.imshow('Prueba', img)

cv2.waitKey()

print("Code Complete")