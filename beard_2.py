#imports
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


beard_cascade = cv.CascadeClassifier('cascade3.xml')
#beard_cascade = cv.CascadeClassifier('cascade(4).xml')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#setting up camera to capture
cap = cv.VideoCapture(0)

while 1:
    ret, img = cap.read()

    beard_img = img.copy()

    gray = cv.cvtColor(beard_img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv.rectangle(beard_img,(x,y),(x+w,y+h),(255,0,0),2)
        img_gray = gray[y-75:y+h+75, x-75:x+w+75]
        img_color = img[y-75:y+h+75, x-75:x+w+75]

        beard_rect = beard_cascade.detectMultiScale(img_gray, 1.3, 5)
        for (bx, by, bw, bh) in beard_rect:
            cv.rectangle(img_color, (bx, by), (bx + bw, by + bh), (0, 0, 0), 3)
            cv.putText(img_color, 'Beard', (bx, by - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv.imshow('face', beard_img)
    cv.imshow('beard', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break






    #beardimg = cv.cvtColor(beardframe, cv.COLOR_BGR2RGB)

cap.release()
cv.destroyAllWindows()

#convert camera frame from bgr to rgb
frame_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# frame_rgb = cv.cvtColor(beard_img, cv.COLOR_GRAY2RGB)
# show camera frame
plt.imshow(frame_rgb)
plt.title('Test frame')
plt.show()


# close camera
cap.release()
